"""
增量重渲(closed_loop §10.1)离线自检 —— 锁住 build_world augment + render_corpus delta 的【无 LLM 路径】。
有 LLM 的部分(augment 真生成新实体 / delta 真渲新实体)靠端到端跑 + 盲审验(no-mock,不在此造假 tracer)。
这里只证:① 不需新实体时 augment 零 LLM 调用 + 旧世界不动;② delta 遇无事实实体早退、旧 docs 原样;
③ 全量路径未被破坏;④ 新关系/事件按字段 owner 触及的旧实体只补精确 session。

跑:./venv/bin/python tests/incremental_render_selftest.py
"""
from __future__ import annotations
import sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.world_state import WorldState, Timeline, Op, SET, _date_of, name_collisions
from pipeline.world_gen import build_world
from pipeline.render import render_corpus
from pipeline.closed_loop import _render_delta_scope

checks: list[tuple[bool, str]] = []


def ck(name, cond):
    checks.append((bool(cond), name))


def _ws(n_ent=2, n_sess=4):
    e = {f"E{i}部": {"f": Timeline([Op(s, _date_of(s), SET, f"v{i}_{s}", None) for s in range(n_sess)])}
         for i in range(n_ent)}
    return WorldState(e, n_sessions=n_sess)


class _BoomTracer:                       # 任何 LLM 调用都炸 —— 证明 no-op 路径零调用
    def chat_json(self, *a, **k):
        raise AssertionError("不该调用 LLM(no-op 路径)")


# ── ① build_world augment:existing 已达 target → 0 新实体、零 LLM、返回既有同对象 ──
ws = _ws(2, 4)
wp = {"domain_profile": {"entity_noun": "部门"}, "shared_world_spec": {"entities": {"count": 2}, "timeline": {"n_sessions": 4}}}
out = build_world(wp, tracer=_BoomTracer(), log=lambda *a: None, existing=ws)
ck("augment:已达 target → 零 LLM 调用(BoomTracer 没炸)", True)   # 跑到这=没炸
ck("augment:返回既有同对象", out is ws)
ck("augment:实体集不变", sorted(out.entities) == ["E0部", "E1部"])
ck("augment:无表面塌缩", name_collisions(out) == [])

# ── ② render_corpus delta:only_entities 指向【无事实】实体 → 每周早退、零 LLM、旧 docs 原样 ──
corpus = {"sessions": [{"session_id": s, "date": _date_of(s),
                        "docs": [{"doc_id": f"s{s}_sig_0", "content": f"E0部本期 v0_{s}"}]} for s in range(4)]}
done = set(range(4))
before = [len(x["docs"]) for x in corpus["sessions"]]
render_corpus({"domain_profile": {}}, ws, 0, tracer=_BoomTracer(), corpus=corpus, done_weeks=done,
              save_cb=lambda: None, log=lambda *a: None, only_entities={"GHOST"})
after = [len(x["docs"]) for x in corpus["sessions"]]
ck("delta:无事实新实体 → 零 LLM(BoomTracer 没炸)", True)
ck("delta:每周 docs 数不变(纯追加、无事实不动)", before == after)
ck("delta:旧 doc_id 原样保留", corpus["sessions"][0]["docs"][0]["doc_id"] == "s0_sig_0")

# ── ③ delta 遍历【所有周】(非只未完成周):done 全满时全量路径会跳过,delta 仍逐周走(只是无事实→不加) ──
#    用一个【有事实】的实体证明 weeks 选择是 all:把 E1部 当"新实体"(它有事实)→ delta 会试图渲它(需 LLM)。
#    这里不真渲(避免 LLM),只验"weeks=all"的选择逻辑:done 全满 + only_entities 给【有事实实体】→ 会进入渲染(故用 Boom 反证它【确实尝试】)。
ws2 = _ws(2, 3)
corpus2 = {"sessions": [{"session_id": s, "date": _date_of(s), "docs": []} for s in range(3)]}
boomed = False
try:
    render_corpus({"domain_profile": {}}, ws2, 0, tracer=_BoomTracer(), corpus=corpus2, done_weeks={0, 1, 2},
                  save_cb=lambda: None, log=lambda *a: None, only_entities={"E1部"})   # E1部 有事实 → 该试图渲 → Boom
except AssertionError:
    boomed = True
ck("delta:done 全满仍遍历所有周(有事实实体确被尝试渲染,证明 weeks=all)", boomed)

# ── ④ typed 结构增量范围:新实体全程，旧实体只补关系/事件实际改变的 session ──
ws3 = _ws(4, 6)
ws3.entity_types = {"E0部": "source", "E1部": "source", "E2部": "target", "E3部": "source"}
ws3.world_blueprint = {
    "entity_types": [
        {"id": "source", "fields": [{"name": "source_ref"}]},
        {"id": "target", "fields": [{"name": "target_ref"}]},
        # 与关系无关的第三类型故意同名；owner 必须只在 relation 两端局部推断。
        {"id": "observer", "fields": [{"name": "target_ref"}]},
    ],
    "relation_types": [
        {"id": "source_owned", "from_type": "source", "to_type": "target", "field": "source_ref"},
        {"id": "target_owned", "from_type": "source", "to_type": "target", "field": "target_ref"},
        {"id": "self_owned", "from_type": "source", "to_type": "source", "field": "source_ref"},
    ],
}
ws3.relations = [
    # 已渲过的旧关系不能再次进入 delta。
    {"id": "rel-old", "type": "source_owned", "from": "E0部", "to": "E2部", "session": 1},
    # 新关系改变旧 source 的 FK，只需补 E1部@4。
    {"id": "rel-new-old-source", "type": "source_owned", "from": "E1部", "to": "E2部", "session": 4},
    # 字段唯一归属 target：即使 source E3部是新实体，也必须补真正被写 Timeline 的旧 E2部@2。
    {"id": "rel-new-new-source", "type": "target_owned", "from": "E3部", "to": "E2部", "session": 2},
    # self-relation 两端同型时固定由 source 持有字段，只补 E1部@2，不能误补 E0部@2。
    {"id": "rel-new-self", "type": "self_owned", "from": "E1部", "to": "E0部", "session": 2},
]
ws3.events = [
    # 已渲过的旧事件不能再次进入 delta。
    {"id": "evt-old", "type": "upgrade", "session": 1,
     "effects": [{"entity": "E2部", "field": "f", "set": "old"}]},
    # 同一新事件同时改变旧 E0部和新 E3部；只需为旧实体补 E0部@3。
    {"id": "evt-new-mixed", "type": "upgrade", "session": 3,
     "effects": [{"entity": "E0部", "field": "f", "set": "new-old"},
                 {"entity": "E3部", "field": "f", "set": "new-entity"}]},
    # 另一个旧实体只在 session 5 被事件改变。
    {"id": "evt-new-old", "type": "upgrade", "session": 5,
     "effects": [{"entity": "E2部", "field": "f", "set": "later"}]},
    # 与关系重复触及同一个 entity·session，输出必须去重。
    {"id": "evt-new-dedup", "type": "upgrade", "session": 4,
     "effects": [{"entity": "E1部", "field": "f", "set": "same-session"}]},
]
new_entities, touched_pairs = _render_delta_scope(
    ws3,
    previous_entities={"E0部", "E1部", "E2部"},
    previous_relations={("id", "rel-old")},
    previous_events={("id", "evt-old")},
)
ck("delta scope:只识别真正新增实体并保持确定排序", new_entities == ["E3部"])
ck("delta scope:关系/事件触及的旧实体精确到 session、去重且不带旧结构",
   touched_pairs == [["E0部", 3], ["E1部", 2], ["E1部", 4], ["E2部", 2], ["E2部", 5]])
ck("delta scope:异型关系按字段唯一归属补 target owner", ["E2部", 2] in touched_pairs)
ck("delta scope:无关第三类型同名字段不干扰 relation owner", ["E2部", 2] in touched_pairs)
ck("delta scope:self-relation 固定补 source owner",
   ["E1部", 2] in touched_pairs and ["E0部", 2] not in touched_pairs)
ck("delta scope:新实体不重复进入 old-entity 精准 pair",
   all(pair[0] != "E3部" for pair in touched_pairs))

# ── ⑤ release receipt 对增量扩容的约束（固定桩仅模拟审阅通过，不证明模型能力） ──
from pipeline.corpus_contract import review_documents, attach_receipts, validate_corpus, fidelity_requirements
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_fixture_helpers import fixed_positive_review


class _ReviewPass:
    def chat_json(self, step, *args, **kwargs):
        assert step == "corpus.review"
        return fixed_positive_review(args[0])


receipt_ws = _ws(1, 2)
receipt_corpus = {"sessions": []}
for sid in range(2):
    documents = [{"doc_id": f"receipt-{sid}", "title": "当期记录",
                  "content": f"E0部本期f为v0_{sid}。"}]
    report = review_documents(_ReviewPass(), receipt_ws, sid, documents, requirements=fidelity_requirements(receipt_ws, sid))
    attach_receipts(documents, report, sid)
    receipt_corpus["sessions"].append({"session_id": sid, "date": _date_of(sid), "docs": documents})
ck("receipt:同一世界的固定审阅样例可验证", validate_corpus(receipt_ws, receipt_corpus)["status"] == "passed")
untouched = deepcopy(receipt_corpus)
render_corpus({"domain_profile": {}, "quality_contract": {"corpus_review": True}},
              receipt_ws, 0, tracer=_BoomTracer(), corpus=receipt_corpus, done_weeks={0, 1},
              save_cb=lambda: None, log=lambda *a: None, only_entities={"GHOST"})
ck("receipt:同世界无事实增量保留旧正文与审阅记录", receipt_corpus == untouched)
ck("receipt:同世界增量后审阅仍有效", validate_corpus(receipt_ws, receipt_corpus)["status"] == "passed")
expanded_ws = deepcopy(receipt_ws)
expanded_ws.entities["新部门"] = {"f": Timeline([Op(0, _date_of(0), SET, "新增值", None)])}
expanded_review = validate_corpus(expanded_ws, receipt_corpus)
stale_ids = {issue.get("doc_id") for issue in expanded_review["issues"]
             if issue.get("code") == "missing_or_stale_document_review"}
ck("receipt:扩容不能静默复用旧完整context审阅", stale_ids == {"receipt-0", "receipt-1"})
ck("receipt:检测扩容失效不改写旧语料", receipt_corpus == untouched)

npass = sum(1 for ok, _ in checks if ok)
for ok, name in checks:
    if not ok:
        print(f"  ✗ {name}")
print(f"[incremental_render self-test] {npass}/{len(checks)} PASS")
sys.exit(0 if npass == len(checks) else 1)
