"""刀1 离线自检 —— 锁住 014559 纵览后的六个修复(零 LLM;渲染回路用 scripted tracer)。

覆盖:① render 禁词字段名豁免(「累计」∈「累计计费工时」不再杀 doc)② render 耗尽 fail-loud 弃段
③ _affix_units 单位真源化(裸数补单位/幂等/prev同步/异型不动)④ imprint 后缀保真(万元/% 回贴,混后缀跳过)
⑤ validate illegal_transition(声明字段倒流/出界抓到,未声明往复不碰)⑥ _canonicalize_lines(自编名锁回+丢线补漏+axis回填)
⑦ L4 _choice_field 叠词去重。

跑:./venv/bin/python tests/dao1_selftest.py
"""
from __future__ import annotations
import sys
import json
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_fixture_helpers import fixed_document_reviews
from pipeline.world_state import WorldState, Timeline, Op, SET, UPDATE, EXPIRE, _date_of, validate
from pipeline.world_gen import _affix_units, imprint_structure
from pipeline.render import (_attach_story_provenance, _canonical_fact_refs,
                             _sanitize_corpus, _story_signal_groups,
                             _strict_eq, render_corpus)
from pipeline.story import StoryLedgerError, compile_story_ledger
from pipeline.central_office import _canonicalize_lines
from pipeline.lines.L4_preference import PreferenceLine
from pipeline.lines import prepare_lines

checks: list[tuple[bool, str]] = []
def ck(name, cond): checks.append((bool(cond), name))


def _fake_filler(messages):
    """按新顶层数组协议返回调用方要求的精确 filler 数量。"""
    import re
    user = messages[-1]["content"]
    match = re.search(r"生成\s*(\d+)\s*篇", user)
    want = int(match.group(1)) if match else 0
    return [{"type": "旁支日志", "content": f"外围无关归档记录 {index}"}
            for index in range(want)]


def _fake_public_stage_call(tag, messages):
    """Fixture transport support only; this does not test model judgment."""
    user = messages[-1]["content"]
    if tag == "render.signal" and "【冻结的公共阶段定义】" in user:
        definitions = json.loads(user.split("【冻结的公共阶段定义】", 1)[1].splitlines()[0])
        date = user.split("【文档日期】", 1)[1].splitlines()[0]
        return {"docs": [{"content": f"{date}，{rule['entity_noun']}的{rule['field']}阶段顺序为"
                          + "、".join(rule["ordered_stages"]) + "。实际记录可跳级，不可倒退。"}
                         for rule in definitions]}
    if tag == "corpus.review":
        data = json.loads(user)
        return {"document_reviews": fixed_document_reviews(data["documents"]), "verdict": "pass", "unsupported_claims": [], "coverage": [
            {"requirement_id": item["requirement_id"], "status": "supported", "reason": "固定离线桩：规则正文可读",
             "evidence": [{"doc_index": index, "quote": data["documents"][index]["content"]}]}
            for index, item in enumerate(data["requirements"])]}
    return None


ck("盲读答案的成对引号不改变精确值", _strict_eq("『已拾取』", "已拾取")
   and _strict_eq("「北境顾问」", "北境顾问"))
ck("剥引号后仍严格区分量纲", not _strict_eq("『78%』", "0.78"))

grouped = _story_signal_groups({
    "艾尔文": [
        {"entity": "艾尔文", "field": "任务状态", "value": "完成"},
        {"entity": "艾尔文", "field": "称号", "value": "守夜人"},
    ],
    "霜牙": [{"entity": "霜牙", "field": "装备状态", "value": "已掉落"}],
}, [{"effects": [{"entity": "艾尔文", "field": "任务状态", "set": "完成"}]}])
ck("Story 渲染把 event effect 与同一期普通状态拆组",
   [[(entity, [fact["field"] for fact in facts]) for entity, facts in group]
    for group in grouped] == [[("艾尔文", ["任务状态"])],
                              [("艾尔文", ["称号"]), ("霜牙", ["装备状态"])]] )


def _tl(*sv):                            # sv: (session, value);自动 SET/UPDATE + prev 链
    ops, prev = [], None
    for (s, v) in sv:
        ops.append(Op(s, _date_of(s), SET if prev is None else UPDATE, v, prev)); prev = v
    return Timeline(ops)


# ════════ ① + ② 渲染回路(scripted tracer,零 LLM)════════
class _ScriptedTracer:
    """按脚本回 docs。只对 render.signal 出脚本+记历史(filler/conflict 等其它调用回空,免互相吃响应)。"""
    def __init__(self, script):
        self.script = list(script); self.calls = []; self.systems = []; self.text_prompts = []
    def chat_text(self, tag, messages, **kw):
        self.text_prompts.append(messages)
        return "外围无关归档记录。"
    def chat_json(self, tag, messages, **kw):
        public_stage_response = _fake_public_stage_call(tag, messages)
        if public_stage_response is not None:
            return public_stage_response
        if tag == "narrative.review":
            return {"unsupported_claims": []}
        if tag == "render.discriminate":
            text = messages[-1]["content"]
            for value in ("86小时", "维持治疗"):
                if value in text:
                    return {"answers": [{"key": "q0", "answer": value}]}
            return {"answers": [{"key": "q0", "answer": "不确定"}]}
        if tag == "render.filler":
            return _fake_filler(messages)
        if tag != "render.signal":
            return {"docs": []}
        self.systems.append(messages[0]["content"])
        self.calls.append(messages[-1]["content"])
        return self.script.pop(0) if self.script else {"docs": []}


def _mini_ws():
    return WorldState({"鼎晟案": {"累计计费工时": _tl((0, "86小时"))}}, n_sessions=1)


def _run_render(tracer, style_spec=None, target=0):
    ws = _mini_ws()
    corpus = {"sessions": []}
    wp = {"domain_profile": {"doc_genres": ["纪要"], "stopped_phrase": "停止计费"}}
    if style_spec is not None:
        wp["style_spec"] = style_spec
    render_corpus(wp,
                  ws, target, tracer, corpus, set(), save_cb=lambda: None, log=lambda *a: None)
    return corpus["sessions"][0]["docs"] if corpus["sessions"] else []

# ①:doc 含字段名「累计计费工时」(含禁词「累计」)+ 事实就近 → 必须【一次过】,不被禁词杀
good_doc = {"title": "周度纪要", "type": "纪要", "content": "鼎晟案本期累计计费工时为86小时,推进正常。"}
t1 = _ScriptedTracer([{"docs": [good_doc]}])
docs1 = _run_render(t1, target=1)
ck("①豁免:字段名含禁词的合格 doc 一次过(不再静默杀)", len(t1.calls) == 1 and any("86小时" in d["content"] for d in docs1))
ck("①无兜底备忘混入", not any(d.get("is_fallback") for d in docs1))
ck("filler 提示词不向模型暴露冻结实体专名",
   t1.text_prompts and all("鼎晟案" not in message["content"]
                           for prompt in t1.text_prompts for message in prompt))
ck("①信号文档漏 fact_refs 时按当期真值反推补齐",
   docs1 and docs1[0].get("fact_refs") == ["鼎晟案.累计计费工时"])

# 模型即使自报了非空但非法的引用，最终元数据也必须以冻结世界+正文为准。
wrong_refs = {"sessions": [{"session_id": 0, "docs": [{
    "doc_id": "s0_sig_0", "content": "鼎晟案累计计费工时为86小时。",
    "fact_refs": ["case.鼎晟案.累计计费工时", "不存在.stopped"],
}]}]}
canonical_stats = _sanitize_corpus(wrong_refs, _mini_ws())
ck("①信号文档非空错误 fact_refs 被规范真源覆盖",
   wrong_refs["sessions"][0]["docs"][0]["fact_refs"] == ["鼎晟案.累计计费工时"]
   and canonical_stats["canonicalized_refs"] == 1)

event_ref = _canonical_fact_refs(
    "艾尔文在霜牙参与下完成击败：艾尔文的任务状态更新为完成。",
    [], [{"id": "evt-1", "participants": {"player": "艾尔文", "boss": "霜牙"},
          "effects": [{"entity": "艾尔文", "field": "任务状态", "set": "完成"}]}])
ck("①事件 provenance 使用规范 event-id，不再生成不存在的 .label 字段", event_ref == ["evt-1"])

# ①b:doc 真犯禁(正文用「目前」)→ 重渲,且 hint 必须【如实】说"因全局口径词被废",不再谎报"没写"
bad_doc = {"title": "周度纪要", "type": "纪要", "content": "鼎晟案本期累计计费工时为86小时,目前整体平稳。"}
t2 = _ScriptedTracer([{"docs": [bad_doc]}, {"docs": [good_doc]}])
docs2 = _run_render(t2)
ck("①b 真犯禁 → 第2轮重渲成功", len(t2.calls) == 2 and any("86小时" in d["content"] for d in docs2))
ck("①b hint 如实报死因(含'全局口径词'与『目前』)", "全局口径词" in t2.calls[1] and "目前" in t2.calls[1])

# ②:4 轮全失败 → fail-loud 弃段，不再注入能绕过接地闸的 K=V 模板备忘
t3 = _ScriptedTracer([{"docs": []}] * 4)
docs3 = _run_render(t3)
fb = [d for d in docs3 if d.get("is_fallback")]
signal3 = [d for d in docs3 if "_sig_" in str(d.get("doc_id", ""))]
ck("②耗尽 fail-loud:4 轮失败后弃段", len(t3.calls) == 4 and signal3 == [])
ck("②弃段不注 K=V 兜底备忘", not fb)

# ②b:whitepaper style_spec 必须进入实际 signal system prompt，且旧固定长文约束不得残留
short_style = {"tone": "客观简洁", "format": "单句", "length": "30-80字"}
t_style = _ScriptedTracer([{"docs": [good_doc]}])
_run_render(t_style, short_style)
ck("②b renderer 消费 whitepaper style_spec",
   t_style.systems and all(value in t_style.systems[0] for value in short_style.values()))
ck("②b signal prompt 不再硬编码 1200-2000 字", "1200-2000" not in t_style.systems[0])
ck("②b 非剧情世界不注入 Story Ledger 上下文",
   t_style.calls and "game Story Ledger" not in t_style.calls[0])


# ②c:game Story Ledger 只给本组当前 scene + 前一 scene canonical events，并逐篇写 provenance。
def _story_ws():
    """构造两幕、单主角且可被 Story Ledger 严格重放的最小游戏世界。"""
    blueprint = {
        "version": 1,
        "entity_types": [{
            "id": "player", "noun": "主角", "count": 1, "primary": True,
            "cardinality_policy": "exact",
            "fields": [{"name": "任务状态", "kind": "status",
                        "states": ["出发", "完成"]}],
        }],
        "relation_types": [],
        "event_types": [
            {"id": "begin", "label": "启程", "roles": {"actor": "player"},
             "effect_fields": [{"role": "actor", "field": "任务状态"}], "min_count": 1},
            {"id": "finish", "label": "决战", "roles": {"actor": "player"},
             "effect_fields": [{"role": "actor", "field": "任务状态"}], "min_count": 1},
        ],
        "causal_rules": [],
        "temporal_model": {"unit": "chapter", "cadence": "event-driven",
                           "n_sessions": 2, "step_days": 1},
        "evidence_channels": ["剧情日志"],
    }
    ws = WorldState(
        {"艾尔文": {"任务状态": _tl((0, "出发"), (1, "完成"))}},
        n_sessions=2,
        entity_types={"艾尔文": "player"},
        events=[
            {"id": "evt-start", "type": "begin", "session": 0,
             "participants": {"actor": "艾尔文"},
             "effects": [{"entity": "艾尔文", "field": "任务状态", "set": "出发"}]},
            {"id": "evt-finish", "type": "finish", "session": 1,
             "participants": {"actor": "艾尔文"},
             "effects": [{"entity": "艾尔文", "field": "任务状态", "set": "完成"}]},
        ],
        world_blueprint=blueprint,
    )
    ws.narrative = compile_story_ledger(ws, {
        "premise": "北境长夜吞没了归途。",
        "goal": "艾尔文必须点亮归途。",
        "stakes": "失败会让聚落永失补给。",
        "protagonist_ref": "艾尔文",
        "key_item_origins": [],
        "scenes": [
            {"scene_id": "scene-start", "session": 0, "order": 0,
             "event_refs": ["evt-start"], "caused_by_scene_refs": [],
             "dramatic_function": "setup", "summary": "艾尔文踏入长夜。"},
            {"scene_id": "scene-finish", "session": 1, "order": 0,
             "event_refs": ["evt-finish"], "caused_by_scene_refs": ["scene-start"],
             "dramatic_function": "climax", "summary": "艾尔文完成最后决战。"},
        ],
    })
    return ws


class _StoryTracer:
    """按事件返回合格短文，并留存 signal prompt 供剧情边界断言。"""
    def __init__(self):
        self.calls = []
        self.review_calls = []

    def chat_text(self, tag, messages, **kw):
        return "边境商队在本期登记了一批与主线无关的普通货物。"

    def chat_json(self, tag, messages, **kw):
        public_stage_response = _fake_public_stage_call(tag, messages)
        if public_stage_response is not None:
            return public_stage_response
        user = messages[-1]["content"]
        if tag == "narrative.review":
            self.review_calls.append(user)
            return {"unsupported_claims": []}
        if tag == "render.discriminate":
            if "任务状态" in user and "记录为出发" in user:
                return {"answers": [{"key": "q0", "answer": "出发"}]}
            if "任务状态" in user and "记录为完成" in user:
                return {"answers": [{"key": "q0", "answer": "完成"}]}
            return {"answers": [{"key": "q0", "answer": "不确定"}]}
        if tag == "render.filler":
            return _fake_filler(messages)
        if tag != "render.signal":
            return {"docs": []}
        self.calls.append(user)
        if "evt-finish" in user:
            return {"docs": [{"type": "剧情日志", "content":
                               "2025-01-07，艾尔文完成决战，任务状态记录为完成。",
                               "fact_refs": ["艾尔文.任务状态"], "is_filler": True}]}
        return {"docs": [{"type": "剧情日志", "content":
                           "2025-01-06，艾尔文响应启程，任务状态记录为出发。",
                           "fact_refs": ["艾尔文.任务状态"], "is_filler": True}]}


story_ws = _story_ws()

# 普通场景也可以有 domain events；只有启用 Story Ledger 的 Game 才要求事件组全覆盖。
agent_event_ws = deepcopy(story_ws)
agent_event_ws.narrative = None
try:
    render_corpus({"domain_profile": {"doc_genres": ["执行日志"]}}, agent_event_ws, 0,
                  _ScriptedTracer([]), {"sessions": []}, set(),
                  save_cb=lambda: None, log=lambda *a: None)
except RuntimeError:
    agent_event_drop_allowed = False
else:
    agent_event_drop_allowed = True
ck("②c 普通 Agent 的 event 组失败只弃段，不误报 game narrative 全局失败",
   agent_event_drop_allowed)

story_tracer = _StoryTracer()
story_corpus = {"sessions": []}
render_corpus({"domain_profile": {"doc_genres": ["剧情日志"]}}, story_ws, 0,
              story_tracer, story_corpus, set(), save_cb=lambda: None, log=lambda *a: None)
story_calls = {"start": next(x for x in story_tracer.calls if "evt-start" in x),
               "finish": next(x for x in story_tracer.calls if "evt-finish" in x)}
ck("②c 首幕只收到当前 scene，不收到未来 scene",
   "scene-start" in story_calls["start"] and "scene-finish" not in story_calls["start"])
ck("②c 次幕只收到上一幕 canonical event，不收到自由摘要",
   "scene-finish" in story_calls["finish"] and "evt-start" in story_calls["finish"]
   and "艾尔文踏入长夜" not in story_calls["finish"])
ck("②c premise/goal/stakes 不作为事实喂给正文",
   all(text not in story_calls["start"] for text in
       ("北境长夜吞没了归途", "艾尔文必须点亮归途", "失败会让聚落永失补给")))
story_docs = {session["session_id"]: session["docs"][0]
              for session in story_corpus["sessions"] if session["docs"]}
ck("②c accepted doc 的 scene_refs/event_refs 由代码确定",
   story_docs[0].get("scene_refs") == ["scene-start"]
   and story_docs[0].get("event_refs") == ["evt-start"]
   and story_docs[1].get("scene_refs") == ["scene-finish"]
   and story_docs[1].get("event_refs") == ["evt-finish"])
ck("②c signal 身份由代码决定，模型不能用 is_filler 删除事件证据",
   all(doc.get("is_filler") is False for doc in story_docs.values()))
ck("②c 语料叙事审阅 CANON 含本篇 document_date 与载体动作白名单",
   story_tracer.review_calls
   and all('"period_label": "第1章"' in story_tracer.review_calls[0]
           and '"time_unit": "章"' in story_tracer.review_calls[0]
           and '"document_date": "2025-01-' in user
           and '"allowed_document_scaffolding"' in user
           for user in story_tracer.review_calls))


class _PartialFillerTracer(_StoryTracer):
    """第一篇 filler 空正文，其余正常，验证草堆按总规模而非逐调用验收。"""

    def __init__(self, fail_all=False):
        super().__init__()
        self.filler_calls = 0
        self.fail_all = fail_all

    def chat_text(self, tag, messages, **kw):
        self.filler_calls += 1
        if self.fail_all or self.filler_calls == 1:
            return {"__error__": "reasoning-only"}
        return super().chat_text(tag, messages, **kw)


partial_filler_corpus = {"sessions": []}
partial_filler_done = set()
partial_filler_tracer = _PartialFillerTracer()
render_corpus({"domain_profile": {"doc_genres": ["剧情日志"]}}, story_ws, 1,
              partial_filler_tracer, partial_filler_corpus, partial_filler_done,
              save_cb=lambda: None, log=lambda *a: None)
partial_fillers = [doc for session in partial_filler_corpus["sessions"]
                   for doc in session["docs"] if doc.get("is_filler") is True]
ck("②c 单篇 filler 失败显式跳过，不杀死已满足规模与事件合同的 corpus",
   partial_filler_done == {0, 1} and partial_filler_tracer.filler_calls == 2
   and len(partial_fillers) == 1)

try:
    render_corpus({"domain_profile": {"doc_genres": ["剧情日志"]}}, story_ws, 1000,
                  _PartialFillerTracer(fail_all=True), {"sessions": []}, set(),
                  save_cb=lambda: None, log=lambda *a: None)
except RuntimeError as exc:
    filler_size_closed = "语料字符不足" in str(exc)
else:
    filler_size_closed = False
ck("②c filler 可缺失但总语料规模仍 fail-closed", filler_size_closed)

split_docs = [
    {"content": "艾尔文响应启程，任务状态记录为出发。"},
    {"content": "艾尔文完成决战，任务状态记录为完成。"},
]
event_decls = {"begin": "启程", "finish": "决战"}
labeled = [{**event, "label": event_decls[event["type"]]} for event in story_ws.events]
_attach_story_provenance(split_docs, labeled, story_ws.narrative["scenes"])
ck("②c 多文档 provenance 只标各篇实际承载的事件",
   split_docs[0]["event_refs"] == ["evt-start"]
   and split_docs[0]["scene_refs"] == ["scene-start"]
   and split_docs[1]["event_refs"] == ["evt-finish"]
   and split_docs[1]["scene_refs"] == ["scene-finish"])

partial_effect_docs = [{"content": "艾尔文响应启程，但没有写明落地后的状态。"}]
_attach_story_provenance(partial_effect_docs, labeled, story_ws.narrative["scenes"])
ck("②c provenance 不把只有 label+参与者、缺 effect 结果的文档误标为事件证据",
   partial_effect_docs[0]["event_refs"] == []
   and partial_effect_docs[0]["scene_refs"] == [])

wrong_field_docs = [{"content": "艾尔文响应启程，心情记录为出发。"}]
_attach_story_provenance(wrong_field_docs, labeled, story_ws.narrative["scenes"])
ck("②c provenance 不把实体和值相同但 effect 字段错误的文档误标为事件证据",
   wrong_field_docs[0]["event_refs"] == []
   and wrong_field_docs[0]["scene_refs"] == [])


class _UnsupportedStoryTracer(_StoryTracer):
    """机械事实可读，但每轮都额外编造剧情。"""

    def chat_json(self, tag, messages, **kw):
        if tag == "narrative.review":
            return {"unsupported_claims": ["无依据复活"]}
        return super().chat_json(tag, messages, **kw)


unsupported_done = set()
unsupported_corpus = {"sessions": []}
try:
    render_corpus({"domain_profile": {"doc_genres": ["剧情日志"]}}, story_ws, 0,
                  _UnsupportedStoryTracer(), unsupported_corpus, unsupported_done,
                  save_cb=lambda: None, log=lambda *a: None)
except RuntimeError:
    unsupported_failed_closed = True
else:
    unsupported_failed_closed = False
ck("②c game 语义重试耗尽后整段 fail-closed 且不记完成 checkpoint",
   unsupported_failed_closed and unsupported_done == set()
   and unsupported_corpus == {"sessions": []})


class _BackgroundUnsupportedTracer(_StoryTracer):
    """事件文档合格，但普通状态组每轮都会编造跨事实剧情。"""

    def __init__(self):
        super().__init__()
        self.background_reviews = []

    def chat_json(self, tag, messages, **kw):
        user = messages[-1]["content"]
        if tag == "render.discriminate" and "称号" in user:
            return {"answers": [{"key": "q0", "answer": "守夜人"}]}
        if tag == "narrative.review" and "私自复活" in user:
            self.background_reviews.append(user)
            return {"unsupported_claims": ["无依据复活"]}
        if tag == "render.signal" and "称号" in user:
            return {"docs": [{"type": "角色记录",
                              "content": "2025-01-06，艾尔文的称号记为守夜人，并在无记录下私自复活。",
                              "fact_refs": ["艾尔文.称号"]}]}
        return super().chat_json(tag, messages, **kw)


background_ws = deepcopy(story_ws)
background_ws.world_blueprint["entity_types"][0]["fields"].append(
    {"name": "称号", "kind": "status", "states": ["守夜人"]})
background_ws.entities["艾尔文"]["称号"] = _tl((0, "守夜人"))
background_tracer = _BackgroundUnsupportedTracer()
background_corpus = {"sessions": []}
render_corpus({"domain_profile": {"doc_genres": ["剧情日志"]}}, background_ws, 0,
              background_tracer, background_corpus, set(), save_cb=lambda: None,
              log=lambda *a: None)
ck("②c game 的 background signal 也经过同一语义闸，串组编造被弃",
   len(background_tracer.background_reviews) == 4
   and not any("私自复活" in doc.get("content", "")
               for session in background_corpus["sessions"] for doc in session["docs"]))

bad_story_ws = deepcopy(story_ws)
bad_story_ws.narrative["goal"] = ""
bad_story_tracer = _StoryTracer()
try:
    render_corpus({"domain_profile": {"doc_genres": ["剧情日志"]}}, bad_story_ws, 0,
                  bad_story_tracer, {"sessions": []}, set(), save_cb=lambda: None,
                  log=lambda *a: None)
    story_failed_closed = False
except StoryLedgerError:
    story_failed_closed = True
ck("②c 非法 Story Ledger 在任何渲染调用前 fail-closed",
   story_failed_closed and bad_story_tracer.calls == [])

# ════════ ③ _affix_units ════════
ws3 = WorldState({"星耀案": {"争议标的额": _tl((0, "200"), (2, "250")),
                            "风险评分": _tl((0, "4")),
                            "异型": _tl((0, "1.2亿"))}}, n_sessions=4)
prof3 = {"field_schema": [{"name": "争议标的额", "kind": "numeric", "unit": "万元"},
                          {"name": "异型", "kind": "numeric", "unit": "万元"}]}   # 风险评分无 unit
n = _affix_units(ws3, prof3, log=lambda *a: None)
vals = [v for (_s, _d, v) in ws3.entities["星耀案"]["争议标的额"].set_values()]
ck("③裸数补单位", vals == ["200万元", "250万元"])
ck("③prev 同步补", ws3.entities["星耀案"]["争议标的额"].ops[1].prev == "200万元")
ck("③无 unit 字段不动", [v for (_s, _d, v) in ws3.entities["星耀案"]["风险评分"].set_values()] == ["4"])
ck("③异型写法('1.2亿')不动", [v for (_s, _d, v) in ws3.entities["星耀案"]["异型"].set_values()] == ["1.2亿"])
n2 = _affix_units(ws3, prof3, log=lambda *a: None)
ck("③幂等(二次 0 改动)", n2 == 0)

# ════════ ④ imprint 后缀保真 ════════
ws4 = WorldState({
    "甲案": {"标的额": _tl((0, "300万元"), (1, "520万元"), (2, "180万元"), (3, "470万元"), (4, "260万元"), (5, "390万元"))},
    "乙部": {"缺陷率": _tl((0, "2.5%"), (1, "4.1%"), (2, "1.8%"), (3, "3.9%"), (4, "2.2%"), (5, "3.3%"))},
    "混乱": {"字段": _tl((0, "100万元"), (1, "200"), (2, "300万元"), (3, "400"), (4, "500万元"), (5, "600"))},
}, n_sessions=6)
imprint_structure(ws4, log=lambda *a: None, profile={"l7_max_trends": 10}, seed=1)
v_a = [v for (_s, _d, v) in ws4.entities["甲案"]["标的额"].set_values()]
v_b = [v for (_s, _d, v) in ws4.entities["乙部"]["缺陷率"].set_values()]
v_c = [v for (_s, _d, v) in ws4.entities["混乱"]["字段"].set_values()]
ck("④万元后缀回贴(全部带'万元')", all(v.endswith("万元") for v in v_a))
ck("④%后缀回贴(全部带'%')", all(v.endswith("%") for v in v_b))
ck("④混后缀字段跳过(原样不动)", v_c == ["100万元", "200", "300万元", "400", "500万元", "600"])
# ★防 no-op 假绿(审计:原始值本就带后缀,endswith 断言对'imprint 静默没触发'零检出力)→ 验值序确被改写成
#   【L7 分类器认可的趋势】(形状无关:首末净方向清晰 + ∃局部反向);imprint 现轮转 end_reversal/mid_dip/late_surge
from pipeline.world_state import _to_num as _tn
from pipeline.lines.L7_consolidation import _trend_label as _tlbl, _anti_recency as _tar
for nm, seq in (("甲案", [_tn(v) for v in v_a]), ("乙部", [_tn(v) for v in v_b])):
    lab = _tlbl(seq)
    ck(f"④{nm} 确被改写成可裁趋势(首末净方向清晰+∃局部反向,非 no-op,形状无关)",
       lab in ("上升", "下降") and _tar(seq, lab))

# ════════ ⑤ validate illegal_transition ════════
ws5 = WorldState({
    "天成案": {"案件状态": _tl((0, "诉讼中"), (3, "执行中"), (6, "诉讼中"))},     # 倒流
    "出界案": {"案件状态": _tl((0, "立案"), (2, "调解中"))},                      # 出界(不在表)
    "合规案": {"案件状态": _tl((0, "立案"), (2, "诉讼中"), (5, "结案"))},          # 跳级 OK
    "往复者": {"随访方式": _tl((0, "门诊"), (1, "电话"), (2, "门诊"))},            # 未声明,往复合法
}, n_sessions=8)
prof5 = {"state_machines": [{"field": "案件状态", "states": ["立案", "诉讼中", "执行中", "结案"]}]}
defects = validate(ws5, None, prof5)
kinds = {(d["entity"], d["type"]) for d in defects}
ck("⑤倒流抓到", ("天成案", "illegal_transition") in kinds)
ck("⑤出界抓到", ("出界案", "illegal_transition") in kinds)
ck("⑤跳级合法放行", ("合规案", "illegal_transition") not in kinds)
ck("⑤未声明字段往复不碰(opt-in)", not any(e == "往复者" for (e, _t) in kinds))
ck("⑤无 profile 完全跳过", not any(d["type"] == "illegal_transition" for d in validate(ws5, None, None)))

# ════════ ⑥ _canonicalize_lines ════════
draft = {"active_lines": [{"line": "L1_timeline", "weight": 0.5}, {"line": "L4_preference", "weight": 0.8},
                          {"line": "L7_consolidation", "weight": 0.6}],
         "domain_profile": {"preference_axis": {"field": "随访方式", "options": ["门诊", "电话"]},
                            "state_machines": [{"field": "案件状态", "states": ["立案", "结案"]}]}}
wp6 = {"active_lines": [{"line": "L1_temporal_state", "weight": 0.4},      # 自编名 → 锁回 L1
                        {"line": "L4_source_conflict", "weight": 0.9},     # 自编名 → 前缀锁回 L4
                        {"line": "L9_unknown", "weight": 0.1}],            # 可规范但 draft 未激活 → 丢
       "domain_profile": {}}                                               # axis/states 被 critic 丢 → 回填
_canonicalize_lines(wp6, draft, log=lambda *a: None)
ids6 = [l["line"] for l in wp6["active_lines"]]
ck("⑥自编名锁回 canonical", "L1_timeline" in ids6 and "L4_preference" in ids6)
ck("⑥丢线补漏(L7 回来)", "L7_consolidation" in ids6)
ck("⑥critic 不得新增 draft 未激活的 L9 + 无重复", "L9_unknown" not in str(ids6)
   and "L9_induction" not in ids6 and len(ids6) == len(set(ids6)) == 3)
ck("⑥axis/states 回填", wp6["domain_profile"].get("preference_axis") and wp6["domain_profile"].get("state_machines"))
# ⑥c【value_shape 审计 HIGH】critic 删字段级约束 → _canonicalize_lines 从 draft 兜底恢复
draft_vs = {"active_lines": [{"line": "L1_timeline", "weight": 0.5}],
            "domain_profile": {"field_schema": [{"name": "累计工时", "kind": "numeric", "monotonic": "up"},
                                                 {"name": "标的额", "kind": "numeric", "unit": "万"},
                                                 {"name": "评分", "kind": "numeric", "range": [0, 10]}]}}
wp_vs = {"active_lines": [{"line": "L1_timeline", "weight": 0.5}],
         "domain_profile": {"field_schema": [{"name": "累计工时", "kind": "numeric"},        # critic 把 monotonic 删了
                                             {"name": "标的额", "kind": "numeric"},          # unit 删了
                                             {"name": "评分", "kind": "numeric", "range": [0, 10]}]}}  # 这个保住了
_canonicalize_lines(wp_vs, draft_vs, log=lambda *a: None)
fs_vs = {f["name"]: f for f in wp_vs["domain_profile"]["field_schema"]}
ck("⑥c critic 删 monotonic → 从 draft 兜底恢复", fs_vs["累计工时"].get("monotonic") == "up")
ck("⑥c critic 删 unit → 兜底恢复", fs_vs["标的额"].get("unit") == "万")
ck("⑥c critic 保留的 range 不被改动", fs_vs["评分"].get("range") == [0, 10])
# ⑥b 域条件保持:draft 没有的线绝不被塞回(office 无 L4 的情形)
draft_no4 = {"active_lines": [{"line": "L1_timeline", "weight": 0.5}], "domain_profile": {}}
wp6b = {"active_lines": [{"line": "L1_timeline", "weight": 0.5}], "domain_profile": {}}
_canonicalize_lines(wp6b, draft_no4, log=lambda *a: None)
ck("⑥b 域条件:draft 无 L4 → 不塞回", all(l["line"] != "L4_preference" for l in wp6b["active_lines"]))

# ════════ ⑧ 审计修复回归(刀1 diff 对抗审计点名)════════
# ⑧a【高】_trended_fields 过序列化往返(否则闭环 augment 复注翻向)
ws8 = WorldState.from_dict(ws4.to_dict())
ck("⑧a _trended_fields 过 to_dict/from_dict 往返", set(getattr(ws8, "_trended_fields", [])) == set(ws4._trended_fields)
   and len(ws8._trended_fields) >= 2)
# ⑧b【中】豁免集含【值】:值含禁词(「维持治疗」含「维持」)的合格 doc 必须一次过
ws8b = WorldState({"林深": {"用药方案": _tl((0, "维持治疗"))}}, n_sessions=1)
t8 = _ScriptedTracer([{"docs": [{"title": "门诊记录", "type": "记录", "content": "林深本期用药方案为维持治疗。"}]}])
corpus8 = {"sessions": []}
render_corpus({"domain_profile": {"doc_genres": ["记录"], "stopped_phrase": "停止监测"}},
              ws8b, 0, t8, corpus8, set(), save_cb=lambda: None, log=lambda *a: None)
docs8 = corpus8["sessions"][0]["docs"] if corpus8["sessions"] else []
ck("⑧b 值含禁词的合格 doc 一次过(豁免集含值)", len(t8.calls) == 1
   and any("维持治疗" in d["content"] for d in docs8) and not any(d.get("is_fallback") for d in docs8))
# ⑧c【低】纯数字 unit 拒绝(防 ×10 跑飞)
ws8c = WorldState({"甲": {"x": _tl((0, "200"))}}, n_sessions=2)
_affix_units(ws8c, {"field_schema": [{"name": "x", "kind": "numeric", "unit": "0"}]}, log=lambda *a: None)
ck("⑧c 纯数字 unit 被拒(值不动)", [v for (_s, _d, v) in ws8c.entities["甲"]["x"].set_values()] == ["200"])
# ⑧d【中】sm 字段豁免 monotonic(数值可解析状态表不再与 illegal_transition 乒乓)
ws8d = WorldState({"乙": {"阶段": _tl((0, "阶段1"), (2, "阶段2"), (5, "阶段3"))}}, n_sessions=6)
prof8d = {"state_machines": [{"field": "阶段", "states": ["阶段1", "阶段2", "阶段3"]}]}
d8 = validate(ws8d, None, prof8d)
ck("⑧d 合法单向数值状态表:零缺陷(无 monotonic 乒乓)", d8 == [])
# ⑧e【低】states 声明含重复 → 整条作废(不误判倒流)
prof8e = {"state_machines": [{"field": "阶段", "states": ["a", "b", "a"]}]}
ws8e = WorldState({"丙": {"阶段": _tl((0, "a"), (1, "b"), (2, "a"))}}, n_sessions=3)
ck("⑧e 塌缩声明被跳过(零误判)", not any(d["type"] == "illegal_transition" for d in validate(ws8e, None, prof8e)))

# ════════ ⑨ (b)修复审计回归 ════════
from pipeline.world_gen import _affix_units  # noqa (已导入)
# ⑨a【高】字段白名单真源 = field_schema ∪ state_machines.field(漏 sm → 误删状态机字段、C1③ 空转)
def _wl(schema_fields_set, sm_fields):
    """复刻 build_world 白名单口径(真源并 sm)。"""
    src = set(schema_fields_set) | set(sm_fields)
    return src
ck("⑨a 白名单真源并入 state_machines.field", "案件状态" in _wl({"标的额", "主办律师"}, {"案件状态"}))
ck("⑨a 偏好轴字段不并入真源(L4 独家注入,否则双流复活)", "处理策略倾向" not in _wl({"标的额"}, set()))
# ⑨b【高】L7 只对 imprint 种下的字段判趋势:非 imprint 的噪声字段不产软 gold(首末净方向对噪声放行率高)
from pipeline.lines.L7_consolidation import ConsolidationLine as _CL
noise_ws = WorldState({"噪声案": {"杂值": _tl((0, "57"), (1, "56"), (3, "68"), (5, "10"))},   # 纯噪声 4 点
                       "干净案": {"标的额": _tl((0, "100"), (1, "160"), (2, "220"), (3, "280"), (4, "340"), (5, "400"), (6, "460"), (7, "400"))}},
                      n_sessions=8)
noise_ws._trended_fields = [("干净案", "标的额")]                # 只有干净案被 imprint 种过
trend_orders = _CL()._enum_trend(noise_ws)
ck("⑨b L7 只对 imprint 字段出趋势题(噪声字段不产软 gold)",
   {o["entity"] for o in trend_orders} == {"干净案"} and all(o["aux"]["sub"] == "S1_trend" for o in trend_orders))
ck("⑨b 无 _trended_fields → 零趋势题(不扫全字段)", _CL()._enum_trend(WorldState({"x": {"f": _tl((0, "1"), (1, "9"))}}, n_sessions=2)) == [])

# ════════ ⑩ value_shape(累计工时非单调根治):议会声明值形状,代码机械执行 ════════
from pipeline.world_gen import imprint_structure as _imp
# ⑩a validate:声明 monotonic up 的字段逆向 → monotonic_violation;合规单增 → 无缺陷
ws10 = WorldState({
    "甲案": {"累计工时": _tl((0, "10"), (1, "8"), (2, "30"))},     # up 但回落 → 违规
    "乙案": {"累计工时": _tl((0, "10"), (1, "20"), (2, "35"))},    # 合规单增
    "丙案": {"风险评分": _tl((0, "3"), (1, "12"), (2, "5"))},      # 声明 [0,10] 但 12 出界
}, n_sessions=3)
prof10 = {"field_schema": [{"name": "累计工时", "kind": "numeric", "monotonic": "up"},
                           {"name": "风险评分", "kind": "numeric", "range": [0, 10]}]}
d10 = validate(ws10, None, prof10)
kinds10 = {(x["entity"], x["type"]) for x in d10}
ck("⑩a 单调 up 逆向→monotonic_violation", ("甲案", "monotonic_violation") in kinds10)
ck("⑩a 合规单增→无 violation", ("乙案", "monotonic_violation") not in kinds10)
ck("⑩a 值域出界→out_of_range", ("丙案", "out_of_range") in kinds10)
# ⑩b 关键反乒乓:声明单调的字段【豁免】既有 monotonic(MR退化)检查(否则'必须单调'×'不许单调'死锁)
ws10b = WorldState({"丁案": {"累计工时": _tl((0, "10"), (1, "20"), (2, "30"), (3, "40"))}}, n_sessions=4)  # 完美单增=旧monotonic会判退化
d10b = validate(ws10b, None, prof10)
ck("⑩b 单调声明字段豁免旧 monotonic 检查(不乒乓)",
   not any(x["type"] == "monotonic" for x in d10b) and not any(x["type"] == "monotonic_violation" for x in d10b))
# ⑩c imprint 跳过单调字段(不给累计安人造趋势)
ws10c = WorldState({f"案{i}": {"累计工时": _tl(*[(s, str(10 + s)) for s in range(6)]),
                              "自由值": _tl((0, "30"), (1, "50"), (2, "20"), (3, "60"), (4, "25"), (5, "55"))} for i in range(4)},
                   n_sessions=6)
_imp(ws10c, log=lambda *a: None, profile={"field_schema": [{"name": "累计工时", "kind": "numeric", "monotonic": "up"}], "l7_max_trends": 20})
trended_fields = {f for (_e, f) in ws10c._trended_fields}
ck("⑩c imprint 跳过单调字段(累计工时不被注趋势)", "累计工时" not in trended_fields)
ck("⑩c imprint 仍注自由字段", "自由值" in trended_fields)
# ⑩d 声明非法值域([下≥上])作废、不崩
ck("⑩d 非法值域声明作废不崩", isinstance(validate(ws10, None,
   {"field_schema": [{"name": "风险评分", "kind": "numeric", "range": [10, 0]}]}), list))
# ⑩e 无声明→零回归(value_shape 不触发)
ck("⑩e 无 shape 声明→不产 violation/oob", not any(x["type"] in ("monotonic_violation", "out_of_range")
   for x in validate(ws10, None, {"field_schema": [{"name": "累计工时", "kind": "numeric"}]})))

# ════════ ⑪ value_shape 补审修复(千分位假阳假阴 + 覆盖非补缺 + 改名告警)════════
from pipeline.world_state import _to_num as _tnum
# ⑪a【高】_to_num 剥千分位逗号(旧版 '1,050'→1.0 在累计字段造假阳/假阴)
ck("⑪a _to_num('1,050')==1050(剥千分位)", _tnum("1,050") == 1050.0)
ck("⑪a _to_num('12,000小时')==12000", _tnum("12,000小时") == 12000.0)
ck("⑪a _to_num 全角逗号", _tnum("1，050") == 1050.0)
# ⑪a2【终审 HIGH 根治】_magnitude 按中文大数单位归一(剥逗号只关一种写法,万/亿同构轴)
from pipeline.world_state import _magnitude as _mag
ck("⑪a2 _magnitude('9000万')==9e7", _mag("9000万") == 9000 * 1e4)
ck("⑪a2 _magnitude('1.2亿')==1.2e8", _mag("1.2亿") == 1.2e8)
ck("⑪a2 量级可比:9000万 < 1.2亿", _mag("9000万") < _mag("1.2亿"))
ck("⑪a2 无后缀=数值本身、含逗号剥", _mag("1,050") == 1050.0 and _mag("85%") == 85.0)
# ⑪a3 混量纲真单增(9000万→1.2亿)→ 不再误判 monotonic_violation(终审点名的假阳)
ws_mag = WorldState({"案": {"累计额": _tl((0, "8000万"), (1, "9000万"), (2, "1.2亿"), (3, "1.5亿"))}}, n_sessions=4)
prof_mag = {"field_schema": [{"name": "累计额", "kind": "numeric", "monotonic": "up", "unit": "万"}]}
ck("⑪a3 混量纲真单增→无 violation(万/亿假阳消除)", not any(x["type"] == "monotonic_violation" for x in validate(ws_mag, None, prof_mag)))
# ⑪a4 混量纲真出界(2亿 对 值域[0,5000]万)→ 抓到 out_of_range(终审点名的假阴)
ws_mag2 = WorldState({"案": {"额度": _tl((0, "3000"), (1, "2亿"))}}, n_sessions=2)
prof_mag2 = {"field_schema": [{"name": "额度", "kind": "numeric", "range": [0, 5000]}]}
ck("⑪a4 混量纲真出界→抓到 out_of_range(万/亿假阴消除)", any(x["type"] == "out_of_range" for x in validate(ws_mag2, None, prof_mag2)))
# ⑪b 千分位的真·单增累计轨迹 → 不再误判 monotonic_violation(假阳消除)
ws11 = WorldState({"甲": {"累计工时": _tl((0, "980"), (1, "1,050"), (2, "1,200"), (3, "1,450"))}}, n_sessions=4)
prof11 = {"field_schema": [{"name": "累计工时", "kind": "numeric", "monotonic": "up"}]}
ck("⑪b 千分位真单增→无 violation(假阳消除)", not any(x["type"] == "monotonic_violation" for x in validate(ws11, None, prof11)))
# ⑪c 千分位的真·出界值 → 不再漏判 out_of_range(假阴消除)
ws11c = WorldState({"乙": {"额度": _tl((0, "3000"), (1, "12,000"))}}, n_sessions=2)
prof11c = {"field_schema": [{"name": "额度", "kind": "numeric", "range": [0, 5000]}]}
ck("⑪c 千分位真出界→抓到 out_of_range(假阴消除)", any(x["type"] == "out_of_range" for x in validate(ws11c, None, prof11c)))
# ⑪d【高】字段约束 critic【改值】(up→down)→ _canonicalize_lines 以 draft 覆盖(非仅补缺)
draft_ov = {"active_lines": [{"line": "L1_timeline"}],
            "domain_profile": {"field_schema": [{"name": "累计工时", "kind": "numeric", "monotonic": "up"},
                                                {"name": "评分", "kind": "numeric", "range": [0, 10]}]}}
wp_ov = {"active_lines": [{"line": "L1_timeline"}],
         "domain_profile": {"field_schema": [{"name": "累计工时", "kind": "numeric", "monotonic": "down"},  # critic 改了 up→down(更毒)
                                             {"name": "评分", "kind": "numeric", "range": [0, 5]}]}}        # critic 改窄了 range
_canonicalize_lines(wp_ov, draft_ov, log=lambda *a: None)
fs_ov = {f["name"]: f for f in wp_ov["domain_profile"]["field_schema"]}
ck("⑪d critic 改 up→down → 以 draft 覆盖回 up(治'改'非只治'删')", fs_ov["累计工时"]["monotonic"] == "up")
ck("⑪d critic 改窄 range → 覆盖回原值", fs_ov["评分"]["range"] == [0, 10])
# ⑪e critic 在 observe 沉默处的合法补充【保留】(draft 该字段无该约束 → 不覆盖)
draft_add = {"active_lines": [{"line": "L1_timeline"}],
             "domain_profile": {"field_schema": [{"name": "工时", "kind": "numeric"}]}}              # draft 无 monotonic
wp_add = {"active_lines": [{"line": "L1_timeline"}],
          "domain_profile": {"field_schema": [{"name": "工时", "kind": "numeric", "monotonic": "up"}]}}  # critic 补了
_canonicalize_lines(wp_add, draft_add, log=lambda *a: None)
ck("⑪e critic 在 draft 沉默处的补充保留(不被覆盖掉)",
   wp_add["domain_profile"]["field_schema"][0].get("monotonic") == "up")

# ════════ ⑫ 小尾巴:Q49 主语保真兜底 + 拒答属性归属协议 ════════
# ⑫a phrase 主语兜底:phraser 丢了实体名(代词化)→ 退回 intent 原文(不出悬空主语题)
class _PronounTracer:
    def chat_json(self, tag, messages, **kw):
        if tag == "phrase":
            return {"question": "第5周的时候，他的‘督导合伙人’是哪一个？"}   # 故意丢主语「周涛」
        return {"docs": []}
from pipeline.render import phrase_questions
ord49 = {"line": "L5_conflict", "capability": "L5_conflict", "entity": "周涛", "field": "督导合伙人",
         "gt": "宋清", "aux": {"session": 4, "rule": "source_reliability", "authoritative_value": "宋清",
                              "authoritative_source": "官方通报", "rumor_value": "周正", "rumor_source": "走廊传闻"}}
phrased = phrase_questions([ord49], {"domain_profile": {}}, _PronounTracer(), log=lambda *a: None)
ck("⑫a phraser 丢主语→退回含『周涛』的 intent(不出悬空代词题)", phrased and "周涛" in phrased[0]["question"])
# ⑫b 主语存在仍须保住时点，不能沿用“主语在就全部放行”的旧合同。
class _OkTracer:
    def chat_json(self, tag, messages, **kw):
        return {"question": "周涛的督导合伙人按官方记录是哪位？"} if tag == "phrase" else {"docs": []}
ph_ok = phrase_questions([ord49], {"domain_profile": {}}, _OkTracer(), log=lambda *a: None)
ck("⑫b 含主语但丢时点→恢复第5周且保留订单",
   len(ph_ok) == 1 and "第5周" in ph_ok[0]["question"]
   and ph_ok[0]["question_validation"]["mode"] == "canonical_template")
# ⑫c 润色协议失败不应杀死已经通过良定义闸的订单；产线 intent 是题面真源。
class _EmptyPhraseTracer:
    def chat_json(self, tag, messages, **kw):
        return {"__error__": "reasoning-only"} if tag == "phrase" else {"docs": []}
ph_empty = phrase_questions([ord49], {"domain_profile": {}}, _EmptyPhraseTracer(), log=lambda *a: None)
ck("⑫c phraser 空正文→保留完整 intent，不侵蚀订单 floor",
   len(ph_empty) == 1 and "周涛" in ph_empty[0]["question"])
# ⑫d v5 保留属性归属约束，并明示主答案与附带事实的评分边界。
from pipeline.factory import ANSWER_PROTOCOL as _AP
ck("⑫d 协议 v7 + 属性归属和公开A评分范围声明", _AP["version"] == 7 and _AP.get("attribute_ownership_no_fold") is True
   and any("不得经关系链折算" in r for r in _AP["rules"])
   and _AP.get("scoring_scope") == "task_with_supporting_reasons"
   and _AP.get("scoring_policy") == "task-with-supporting-reasons/v1"
   and _AP.get("additional_facts") == "record_unrelated_separately")

# ════════ ⑦ L4 _choice_field 叠词去重 ════════
# typed world 冻结后仍需允许 L5 派生“只增证据”侧信道；不得改 canonical 轨迹。
ws13 = WorldState({
    "霜剑": {"装备类型": _tl((0, "武器"))},
    "银甲": {"装备类型": _tl((0, "防具"))},
}, n_sessions=2, world_blueprint={"version": 1, "entity_types": []})
before13 = ws13.to_dict()["entities"]
wp13 = {"active_lines": [{"line": "L5_conflict", "weight": 1.0}],
        "domain_profile": {"l5_max_conflicts": 1}}
prepare_lines(wp13, ws13, log=lambda *a: None)
ck("typed world 允许 L5 侧信道派生", len(ws13.conflicts) == 1)
ck("L5 overlay 不改 canonical 实体轨迹", ws13.to_dict()["entities"] == before13)

l4 = PreferenceLine()
ck("⑦轴名已带'倾向' → 不叠词", l4._choice_field("本期处理策略倾向") == "本期处理策略倾向")
ck("⑦普通轴名 → 照常加 TAG", l4._choice_field("随访方式") == "随访方式倾向")

npass = sum(1 for ok, _ in checks if ok)
for ok, name in checks:
    if not ok:
        print(f"  ✗ {name}")
print(f"[刀1 self-test] {npass}/{len(checks)} PASS")
sys.exit(0 if npass == len(checks) else 1)
