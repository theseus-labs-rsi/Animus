"""
pipeline.render —— 文本渲染层(从 run_factory_v2 拆出,行为不变)。
把结构化产物渲染成自然语言文本:世界事实→语料文档(render_corpus + 助手),订单意图→题面(phrase_questions)。
"""
from __future__ import annotations
import json, re, threading, sys
from math import ceil, isfinite
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 允许 `python pipeline/render.py` 直跑(找到根目录 config)
import config
from pipeline.world_state import _date_of, week_label, _to_num, _dicts, EXPIRE, DELETE
from pipeline.lines import line_for
from pipeline.prompts import render
from pipeline.story import replay_story_ledger, review_narrative_supportedness


LEAK_BANNED = ["当前", "现在", "最新", "目前", "截至目前", "迄今", "至今", "一直", "历来",
               "维持", "保持不变", "累计", "现任", "如今", "始终", "仍为", "仍是", "依旧"]

# 单篇正文不需要 JSON 容器；保留 16384 预算以避免 reasoning 挤空正文。
FILLER_TEXT_MAX_TOKENS = 16_384
DISCRIMINATOR_MAX_TOKENS = 16_384
SIGNAL_MAX_TOKENS = 16_384

# filler 是无关草堆，没有生成凭据形态 token 的业务理由。这里只拦常见、足够长的
# 机器凭据前缀，不尝试做复杂“秘密检测”，避免把普通连字符文本误判。
_CREDENTIAL_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16}|"
    r"gh[pousr]_[A-Za-z0-9]{20,})(?![A-Za-z0-9_-])")

# 盲判别器"不确定/读不出"措辞(判别器明确表达"读不出"→ 该 atom 未忠实渲染)。
_DISC_UNSURE = ["不确定", "无法确定", "读不出", "读不到", "不清楚", "未提及", "未提到",
                "没有提到", "没有提及", "无法判断", "无从", "看不出", "歧义", "不明确",
                "信息不足", "无此", "查无", "找不到", "未找到", "无法读出"]


def _dim_norm(s) -> str:
    """★量纲严格归一(死钉④):去空格、全角数字归一并剥成对引号，保留 %/单位/小数点。
    与 world_state._norm / eval.judge._norm 的关键区别:那两个 .rstrip('%。.') 会把
    "78%"→"78" 抹平双量纲;本函数【保留 %】,使 _dim_norm("78%")!="0.78"、!="78"。"""
    if s is None:
        return ""
    t = str(s).strip().replace(" ", "").replace("　", "")
    t = t.translate(str.maketrans("０１２３４５６７８９％．", "0123456789%."))
    quote_pairs = {"\"": "\"", "'": "'", "“": "”", "‘": "’", "「": "」", "『": "』"}
    while len(t) >= 2 and quote_pairs.get(t[0]) == t[-1]:
        t = t[1:-1].strip()
    return t


def _is_unsure(ans: str) -> bool:
    """判别器是否表达了"读不出/不确定"。"""
    a = _dim_norm(ans).lower()
    if not a:
        return True
    return any(_dim_norm(m).lower() in a for m in _DISC_UNSURE)


def _strict_eq(read: str, gt: str) -> bool:
    """★量纲严格对账(死钉①④):判别器读出的 read 是否严格等于世界真值 gt。
    - 先排除"不确定"类措辞(读不出 → 不算还原)。
    - 量纲敏感:用 _dim_norm(保留 %),"78%" != "0.78"、"78%" != "78"(双量纲必暴露)。
    - 严判 EM(归一后逐字相等),不做子串、不剥 %、不数值近似——任一放宽都会放过双量纲。"""
    if read is None or _is_unsure(read):
        return False
    return _dim_norm(read) == _dim_norm(gt)


def _discriminate_many(docs, queries: list[dict], tracer, *, semantic=False) -> dict[str, str]:
    """派一个盲读者一次回答同组文档的全部 ``entity/field`` 查询。

    查询不含真值；返回值只在调用方与冻结真值逐项严格对账。调用层已经耗尽
    网络/解析重试时立即失败，不能把服务错误伪装成“读不出”后重新渲染正文。
    """
    if not queries:
        return {}
    system = render("discriminate.quality_system" if semantic else "discriminate.system")
    if semantic and any("fact_session" in query for query in queries):
        system += ("\n查询中的 fact_session/fact_date 指事实成立的原时点，不是材料公开日期。"
                   "同一实体字段可以有不同历史时点的多个查询，请按各自时点阅读；"
                   "disclosure_id/target_ref 只用于区分查询，不含答案也不是正文证据。")
    out = tracer.chat_json(
        "render.discriminate",
        [{"role": "system", "content": system},
         {"role": "user", "content": render(
             "discriminate.quality_user" if semantic else "discriminate.user",
             docs="\n\n".join(d for d in docs if d),
             queries=json.dumps(queries, ensure_ascii=False))}],
        temperature=0.0, max_tokens=DISCRIMINATOR_MAX_TOKENS, model=config.DISCRIMINATOR_MODEL,
        **({"response_format": {"type": "json_object"}} if semantic else {}))
    if not isinstance(out, dict) or "__error__" in out:
        detail = out.get("__error__", "非 JSON object") if isinstance(out, dict) else "非 JSON object"
        raise RuntimeError(f"render.discriminate 调用失败:{detail}")
    rows = out.get("answers")
    expected = [str(query.get("key")) for query in queries]
    if not isinstance(rows, list) or any(
            not isinstance(row, dict)
            or row.get("key") is None
            or row.get("answer") is None
            for row in (rows if isinstance(rows, list) else [])):
        raise RuntimeError("render.discriminate 协议失败:answers 必须是完整对象数组")
    keys = [str(row["key"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise RuntimeError(
            f"render.discriminate 协议失败:期望 keys={expected},实际 keys={keys}")
    return {str(row["key"]): str(row["answer"]) for row in rows}


def _discriminator_recovers(docs, entity, field, true_value, tracer):
    """★渲染链第一步核心忠实检:盲读者据 docs 能否唯一、按对量纲还原 (entity,field) 的世界真值?
    返回 (recovered: bool, read_answer: str)。
    - recovered=True  ⟺ 判别器读出的值与 true_value【量纲严格相等】(_strict_eq)= 忠实渲染。
    - recovered=False ⟺ 读不出/读成"不确定"/对不上(含双量纲 78%↔0.78、多跳歧义)= 未忠实渲染 → 进 missing。
    ★死钉①③:true_value 只在【本函数代码侧】用于对账,绝不进判别器 messages(判别器只收 docs+实体+字段)。"""
    ans = _discriminate_many(
        docs, [{"key": "q0", "entity": entity, "field": field}], tracer).get("q0", "")
    return (_strict_eq(ans, str(true_value)), ans)


def _corpus_system(profile, blueprint=None, style_spec=None, *, semantic=False) -> str:
    """构造信号文档提示词，并把白皮书写作规格作为唯一风格约束传入。"""
    genres = "/".join(profile.get("doc_genres", ["周报", "通报", "邮件"]))
    stopped = profile.get("stopped_phrase", "停止统计")
    noun = profile.get("entity_noun", "实体")
    types = (blueprint or {}).get("entity_types") or []
    legend = "、".join(f"{t.get('id')}={t.get('noun')}" for t in types if t.get("id")) or f"legacy={noun}"
    time_unit = ((blueprint or {}).get("temporal_model") or {}).get("unit", "week")
    if isinstance(style_spec, dict):
        style_text = json.dumps(style_spec, ensure_ascii=False)
    elif style_spec:
        style_text = str(style_spec)
    else:
        style_text = "未另行指定；采用该领域真实文档的自然写法，篇幅以完整承载本组事实为准。"
    return render("corpus.quality_system" if semantic else "corpus.system", noun=noun, genres=genres, stopped=stopped,
                  genre0=genres.split("/")[0], type_legend=legend, time_unit=time_unit,
                  style_spec=style_text)


def _filler_system(profile, blueprint=None, corpus_plan=None) -> str:
    """构造同领域、同世界但不承载真值的 filler 提示词。

    输入来自已经冻结的领域画像与世界蓝图；输出只影响草堆文档的风格，
    不允许 filler 接触被追踪实体或字段，因此不会改变 benchmark gold。
    """
    noun = profile.get("entity_noun", "实体")
    corpus_plan = corpus_plan if isinstance(corpus_plan, dict) else {}
    planned_genres = corpus_plan.get("filler_genres")
    planned_genres = planned_genres if isinstance(planned_genres, list) else []
    genres = "/".join(list(dict.fromkeys(
        [item for item in planned_genres + profile.get("doc_genres", ["通知", "纪要", "公告"])
         if isinstance(item, str) and item.strip()])))
    topics = corpus_plan.get("filler_topics")
    topics = topics if isinstance(topics, list) else []
    blueprint = blueprint or {}
    context = {
        "entity_types": [
            {"id": item.get("id"), "noun": item.get("noun")}
            for item in blueprint.get("entity_types", []) if item.get("id")
        ],
        "relation_types": [item.get("id") for item in blueprint.get("relation_types", []) if item.get("id")],
        "event_types": [
            {"id": item.get("id"), "label": item.get("label")}
            for item in blueprint.get("event_types", []) if item.get("id")
        ],
        "evidence_channels": list(blueprint.get("evidence_channels", [])),
    }
    return render("filler.system", noun=noun, genres=genres,
                  filler_topics=json.dumps(topics, ensure_ascii=False),
                  world_context=json.dumps(context, ensure_ascii=False))


def _filler_documents_per_session(wp, target_tokens: int, n_sessions: int) -> int:
    """Keep the historical token floor and add a bounded whitepaper floor."""
    if target_tokens <= 0:
        return 0
    token_floor = max(1, ceil(target_tokens / max(1, n_sessions) / 800))
    plan = wp.get("corpus_plan") if isinstance(wp, dict) else None
    requested = plan.get("filler_documents_per_session") if isinstance(plan, dict) else None
    planned_floor = min(12, max(0, requested)) if type(requested) is int else 0
    return max(token_floor, planned_floor)


def corpus_scale(corpus, target_chars=0, haystack_ratio=None):
    """Measure accepted body characters; this is not a tokenizer estimate."""
    if haystack_ratio is not None and (not isfinite(haystack_ratio) or haystack_ratio < 0):
        raise ValueError("haystack_ratio must be finite and nonnegative")
    docs = [d for s in corpus.get("sessions", []) for d in s.get("docs", [])]
    total = sum(len(d.get("content", "")) for d in docs)
    filler = sum(len(d.get("content", "")) for d in docs if d.get("is_filler") is True)
    rest = total - filler
    required = max(ceil((haystack_ratio or 0) * rest), target_chars - rest, 0)
    return {"unit": "characters", "target_chars": target_chars,
            "haystack_ratio": haystack_ratio, "documents": len(docs),
            "filler_documents": sum(d.get("is_filler") is True for d in docs),
            "total_chars": total, "filler_chars": filler, "other_chars": rest,
            "filler_share": filler / total if total else 0,
            "required_filler_chars": required, "deficit_chars": max(0, required - filler),
            "target_met": filler >= required}


def _top_up_haystack(corpus, target_chars, ratio, tracer, system, blocked,
                    time_unit, save_cb, log=print):
    """Append bounded batches; preserve every accepted sibling before errors."""
    stats = corpus_scale(corpus, target_chars, ratio)
    sessions = corpus.get("sessions", [])
    if stats["target_met"] or not sessions:
        return
    # Initial estimate gives a finite call allowance even for very short output.
    average = stats["filler_chars"] / max(1, stats["filler_documents"])
    estimate = max(400, min(1200, average or 800))
    allowance = 2 * ceil(stats["deficit_chars"] / estimate) + 8
    used, cursor = 0, 0
    known = {str(d.get("doc_id")) for s in sessions for d in s.get("docs", [])}
    while not stats["target_met"] and used < allowance:
        count = min(8, allowance - used, max(1, ceil(stats["deficit_chars"] / estimate)))
        jobs = []
        for _ in range(count):
            session = sessions[cursor % len(sessions)]
            cursor += 1
            sid = session["session_id"]
            slot = len(session.get("docs", [])) + cursor
            jobs.append((session, slot))

        def write_one(job):
            session, slot = job
            try:
                out = tracer.chat_text("render.filler",
                    [{"role": "system", "content": system},
                     {"role": "user", "content": render("filler.user", s=week_label(session["session_id"]),
                                time_unit=time_unit, date=session.get("date", ""), slot=slot)}],
                    temperature=0.9, max_tokens=FILLER_TEXT_MAX_TOKENS)
                # The tracer represents transport/budget failures as records.
                # Stop this pass after saving siblings; retry belongs to resume.
                if isinstance(out, dict) and "__error__" in out:
                    return session, [], RuntimeError(str(out["__error__"]))
                try:
                    return session, _accept_filler_text(out, blocked), None
                except RuntimeError:
                    return session, [], None
            except Exception as exc:
                return session, [], exc

        rows = config.pmap(write_one, jobs, workers=8)
        used += count
        accepted, errors = 0, []
        for session, docs, error in rows:
            if error is not None:
                errors.append(error)
            for doc in docs:
                index = len(session["docs"])
                doc_id = f"s{session['session_id']}_fil_topup_{index}"
                while doc_id in known:
                    index += 1
                    doc_id = f"s{session['session_id']}_fil_topup_{index}"
                known.add(doc_id)
                doc.update(doc_id=doc_id, is_filler=True, fact_refs=[])
                session["docs"].append(doc)
                accepted += 1
        if accepted:
            save_cb()
        stats = corpus_scale(corpus, target_chars, ratio)
        log(f"  草堆补量:本批接受 {accepted}/{count} 篇，总字符 {stats['total_chars']}，"
            f"草堆占比 {stats['filler_share']:.1%}，尚缺 {stats['deficit_chars']} 字")
        if errors:
            raise errors[0]
        if not accepted:
            break
        average = stats["filler_chars"] / max(1, stats["filler_documents"])
        estimate = max(400, min(1200, average or 800))
    if not stats["target_met"]:
        log(f"  ⚠ 语料规模目标未达成，保留已生成正文；欠额 {stats['deficit_chars']} 字，"
            f"本次补量调用 {used}/{allowance}")


def _session_facts(ws, s):
    if getattr(ws, "disclosure", None):
        from pipeline.disclosure import session_facts
        return session_facts(ws, s)
    facts = []
    for ent, flds in ws.entities.items():
        for fname, tl in flds.items():
            same_session = [o for o in tl._sorted() if o.session == s]
            op = same_session[-1] if same_session else None
            if op is None:
                continue
            stopped = op.op in (EXPIRE, DELETE)
            facts.append({"entity": ent, "entity_type": getattr(ws, "entity_types", {}).get(ent),
                          "field": fname, "value": None if stopped else op.value, "stopped": stopped})
    return facts


def _event_is_narrated(event: dict, content: str) -> bool:
    """事件的参与者和全部 effect 三元组同篇出现，才允许挂该事件 provenance。"""
    participants = {str(x) for x in (event.get("participants") or {}).values() if x}
    effects = [effect for effect in (event.get("effects") or []) if isinstance(effect, dict)]

    def _effect_visible(effect: dict) -> bool:
        entity = str(effect.get("entity") or "").strip()
        field = str(effect.get("field") or "").strip()
        value = effect.get("set", effect.get("value"))
        return bool(entity and field and value is not None
                    and entity in content and field in content and str(value) in content)

    return bool(participants and effects
                and all(name in content for name in participants)
                and all(_effect_visible(effect) for effect in effects))


def _missing_event_narratives(events, contents) -> list[str]:
    """机械检查每个领域事件的参与者和 effect 三元组是否完整进入同一篇文档。"""
    missing = []
    for event in events:
        label = str(event.get("label") or "").strip()
        participants = sorted({str(x) for x in (event.get("participants") or {}).values() if x})
        effects = [effect for effect in (event.get("effects") or []) if isinstance(effect, dict)]
        if not label or not participants or not effects:
            missing.append(
                f"事件 {event.get('id') or event.get('type')} 缺 label/participants/effects，无法验叙事")
            continue
        if not any(_event_is_narrated(event, content) for content in contents):
            missing.append(
                f"事件「{label}」必须把参与者 {participants} 与全部 effect 三元组写在同一篇文档中")
    return missing


def _tracked_blocklist(ws, profile=None):
    """草堆禁词表 = 实体专名 + 所有【人名类字段】的取值。

    通用字段词（如“状态”“工具”）不是答案泄漏：没有对应实体专名时无法指向
    benchmark 事实。把它们列为禁词会与“保持同领域”形成不可满足约束。
    ★人名字段从白皮书 `field_schema.kind=="person"` 取(审计 ★1:删掉 '负责/汇报/经理' 中文子串启发式
    —— 那是 office 味、对非 office 域不可靠:medical 的「主治医师/会诊上级」一个 hint 都不匹配)。域知识只从白皮书来。"""
    out = set(ws.entities)
    person_fields = {f.get("name") for f in _dicts((profile or {}).get("field_schema", [])) if f.get("kind") == "person"}
    for flds in ws.entities.values():
        for fname, tl in flds.items():
            if fname in person_fields:
                for (_s, _d, v) in tl.set_values():
                    if v and _to_num(v) is None and len(str(v)) >= 2:
                        out.add(str(v))
    return out


def _accept_filler_text(out, blocked) -> list[dict]:
    """验收单篇 filler 正文；空响应或撞冻结专名时直接失败。"""
    if not isinstance(out, str) or not out.strip():
        detail = (out.get("__error__", f"类型={type(out).__name__}")
                  if isinstance(out, dict) else f"类型={type(out).__name__}")
        raise RuntimeError(f"render.filler 调用/协议失败:{detail}")
    content = out.strip()
    leaks = sorted({str(term) for term in blocked if term and str(term) in content})
    if leaks:
        raise RuntimeError(f"render.filler 命中冻结专名:{leaks}")
    if _CREDENTIAL_TOKEN_RE.search(content):
        raise RuntimeError("render.filler 命中凭据形态 token")
    return [{"type": "背景干扰文档", "content": content}]


def _canonical_fact_refs(content: str, facts: list[dict], events: list[dict]) -> list[str]:
    """从正文与冻结世界反推规范 ``entity.field`` / event-id 引用。"""
    refs: list[str] = []
    for fact in facts:
        entity = str(fact.get("entity") or "")
        field = str(fact.get("field") or "")
        value = fact.get("value")
        value_visible = value not in (None, "") and str(value) in content
        stopped_visible = (fact.get("stopped")
                           and any(word in content for word in ("停止", "不再", "终止", "暂停")))
        if entity and field and entity in content and (value_visible or stopped_visible):
            refs.append(f"{entity}.{field}")
    refs.extend(
        event.get("id") for event in events
        if event.get("id") and _event_is_narrated(event, content)
    )
    return list(dict.fromkeys(ref for ref in refs if ref))


def _sanitize_corpus(corpus: dict, ws, profile=None, *, semantic=False) -> dict:
    """收口语料元数据，尤其处理扩世界后的增量一致性。

    - 所有信号文档的 fact_refs 都从同 session 冻结事实与正文重算；模型自报引用
      不是真源。无法反推则删掉该文档。
    - 扩容后新实体名可能撞上旧 filler，此时删掉撞词 filler，不让草堆变证据。
    """
    from pipeline.corpus_contract import (public_stage_rules, _review_receipt_matches,
                                         authoritative_source_assertions, reviewed_fidelity_provenance)
    public_rules = {rule["rule_id"] for rule in public_stage_rules(ws)}
    blocked = {str(x) for x in _tracked_blocklist(ws, profile) if x}
    events_by_session = {}
    for event in getattr(ws, "events", None) or []:
        events_by_session.setdefault(event.get("session"), []).append(event)
    stats = {"canonicalized_refs": 0, "dropped_unref": 0,
             "dropped_filler_leaks": 0, "dropped_filler_credentials": 0}
    for session in corpus.get("sessions", []):
        sid = session.get("session_id")
        facts = _session_facts(ws, sid)
        kept = []
        for doc in session.get("docs", []):
            content = str(doc.get("content") or "")
            if doc.get("is_filler") is True:
                if _CREDENTIAL_TOKEN_RE.search(content):
                    stats["dropped_filler_credentials"] += 1
                    continue
                if any(term in content for term in blocked):
                    stats["dropped_filler_leaks"] += 1
                    continue
                doc["fact_refs"] = []
            elif "_sig_" in str(doc.get("doc_id", "")):
                if semantic and getattr(ws, "disclosure", None) and not _review_receipt_matches(ws, sid, doc):
                    raise ValueError("Public disclosure material has no current semantic receipt")
                fidelity = reviewed_fidelity_provenance(ws, sid, doc) if semantic else None
                doc["fact_refs"] = (fidelity["fact_refs"] if fidelity is not None else
                                    _canonical_fact_refs(content, facts, events_by_session.get(sid, [])))
                if fidelity is not None:
                    doc["event_refs"] = fidelity["event_refs"]
                rule_refs = doc.get("public_rule_refs", [])
                reviewed_rules = (isinstance(rule_refs, list) and bool(rule_refs)
                                  and all(isinstance(ref, str) for ref in rule_refs)
                                  and set(rule_refs) <= public_rules
                                  and _review_receipt_matches(ws, sid, doc))
                source_refs = doc.get("public_source_refs", [])
                reviewed_sources = (isinstance(source_refs, list) and bool(source_refs)
                                    and all(isinstance(ref, str) for ref in source_refs)
                                    and set(source_refs) <= {item["assertion_id"] for item in
                                                            authoritative_source_assertions(ws, sid)}
                                    and _review_receipt_matches(ws, sid, doc))
                if not doc["fact_refs"] and not reviewed_rules and not reviewed_sources and fidelity is None:
                    stats["dropped_unref"] += 1
                    continue
                stats["canonicalized_refs"] += 1
            kept.append(doc)
        session["docs"] = kept
    return stats


def _render_public_stage_material(wp, ws, tracer, corpus):
    """Render scenario-level definitions inside the original corpus stage.

    The author receives no questions, answers, current entity values or future
    trajectories. Existing reviewed material is reused only when still bound to
    this frozen world. Missing rule material never falls back to a gold snippet.
    """
    from pipeline.corpus_contract import (public_stage_rules, public_rule_coverage_issues,
                                         canonical_context, review_documents, attach_receipts,
                                         CorpusReviewExecutionError)
    rules = public_stage_rules(ws)
    if not rules:
        return False
    problems = public_rule_coverage_issues(ws, corpus)
    if not problems:
        return False
    if any(item["code"] != "missing_public_rule_material" for item in problems):
        raise RuntimeError(f"Existing public stage material is stale or invalid: {problems}")
    missing_ids = {item["rule_id"] for item in problems}
    required = [rule for rule in rules if rule["rule_id"] in missing_ids]
    context = canonical_context(ws, 0)
    date = context["document_date"]
    first = next((session for session in corpus.get("sessions", []) if session.get("session_id") == 0), None)
    if first is None or first.get("date") != date:
        raise RuntimeError("Public stage material requires the correctly dated first corpus period")
    system = render("corpus.public_rules.system",
                    style_spec=json.dumps(wp.get("style_spec") or {}, ensure_ascii=False))
    hint = ""
    for attempt in range(4):
        out = tracer.chat_json("render.signal",
            [{"role": "system", "content": system},
             {"role": "user", "content": render("corpus.public_rules.user", date=date,
                 rules=json.dumps(required, ensure_ascii=False), hint=hint)}],
            temperature=0.6 if attempt == 0 else 0.2, max_tokens=SIGNAL_MAX_TOKENS,
            **({"response_format": {"type": "json_object"}}
               if (wp.get("quality_contract") or {}).get("corpus_review") else {}))
        if not isinstance(out, dict) or "__error__" in out:
            raise RuntimeError("Public stage material author call failed")
        docs = [{**{key: doc[key] for key in ("title", "type", "content") if key in doc},
                 "is_filler": False, "fact_refs": []}
                for doc in _dicts(out.get("docs")) if isinstance(doc.get("content"), str)
                and doc["content"].strip()]
        if not docs:
            raise RuntimeError("Public stage material author returned no document body")
        report = review_documents(tracer, ws, 0, docs, context=context,
                                  required_public_rule_ids=[rule["rule_id"] for rule in required])
        if report["status"] == "error":
            raise CorpusReviewExecutionError(report, 0)
        if report["status"] == "passed":
            attach_receipts(docs, report, 0)
            # A document unreferenced by this complete-body review is not public
            # rule evidence. Do not retain unrelated author additions.
            docs = [doc for doc in docs if doc.get("public_rule_refs")]
            used = {doc.get("doc_id") for session in corpus["sessions"] for doc in session.get("docs", [])}
            index = len(first["docs"])
            for doc in docs:
                while f"s0_sig_{index}" in used:
                    index += 1
                doc["doc_id"] = f"s0_sig_{index}"
                used.add(doc["doc_id"])
                first["docs"].append(doc)
                index += 1
            return True
        hint = "上一版的具体语义缺口，请只据冻结定义修复：" + json.dumps(report["issues"], ensure_ascii=False)
    raise RuntimeError("Public stage material failed its existing corpus review repair budget: " + hint)


def _render_conflict_docs(ws, s, date, _tracer):
    """把冻结的 L5 小道值直接写成低可信文档，确保实体和值逐字可验。"""
    out = []
    for c in (getattr(ws, "conflicts", None) or []):
        if c.get("session") != s or not c.get("rumor_value"):
            continue
        entity = str(c["entity"])
        field = str(c["field"])
        rumor = str(c["rumor_value"])
        source = str(c.get("rumor_source") or "小道消息")
        authority = str(c.get("authoritative_source") or "官方记录")
        content = (
            f"【{source}·未经核实】{date}（第{week_label(s)}{ws.period_unit()}），"
            f"有人声称“{entity}”的“{field}”是“{rumor}”。"
            f"这条说法没有可核验的正式记录，也尚未得到{authority}确认；"
            "现仅按待核实传闻留存，不应当作正式结论。"
        )
        out.append({"type": "待核实传闻", "content": content})
    return out


_SENS_FIELD_LABEL = {"secret": "登录口令", "pii_id": "身份证号", "bankcard": "银行卡号", "apikey": "API 密钥"}


def _render_sensitive_docs(ws, s, date):
    """★L10:把 session==s 的敏感注入渲染成【确定性写入文档】(user 供出 X / assistant 已记录)。
    ★绕开 LLM(不调 tracer):代码直接 Template.substitute → 保证 X 逐字 + 就近实体落地(G3 反退化 L6)。
    无 ws.sensitive(非 L10 场景)→ 返回 [],对其它场景零副作用。"""
    out = []
    for c in (getattr(ws, "sensitive", None) or []):
        if c.get("session") != s or not c.get("value"):
            continue
        label = c.get("field") or _SENS_FIELD_LABEL.get(c.get("stype"), "敏感信息")
        content = render("sensitive.template", date=date, entity=c["entity"],
                         field_label=label, value=c["value"])
        out.append({"type": "记忆写入", "content": content})
    return out


def _render_rule_docs(ws, s, date):
    """★L9:把 session==s 的条件归纳执行实例渲染成【确定性单条情境→动作】文档(用 surface 表面串,canon 层)。
    ★绕开 LLM(不调 tracer):代码直接 Template.substitute → 保证只渲【一条情境+一个处置】、绝不写一般化规则句。
    无 ws.rule_instances(非 L9 场景)→ 返回 [],对其它场景零副作用。"""
    out = []
    for i in (getattr(ws, "rule_instances", None) or []):
        if i.get("session") != s or i.get("surface_action") is None:
            continue
        content = render("rule.template", date=date, inst_id=i.get("inst_id", ""),
                         trigger_field=i.get("trigger_field", ""), x=i.get("x", ""),
                         unit=i.get("unit", ""), surface_action=i.get("surface_action", ""))
        out.append({"type": "处置记录", "content": content})
    return out


def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _story_signal_groups(bysku: dict, events: list[dict]) -> list[list[tuple[str, list[dict]]]]:
    """把事件 effect 事实与同一期普通状态拆组，避免整张角色表拖累主线场景。"""
    effect_keys = {
        (effect.get("entity"), effect.get("field"))
        for event in events for effect in (event.get("effects") or [])
        if isinstance(effect, dict)
    }
    story_group, background = [], []
    for entity, facts in bysku.items():
        event_facts = [fact for fact in facts
                       if (fact.get("entity"), fact.get("field")) in effect_keys]
        other_facts = [fact for fact in facts
                       if (fact.get("entity"), fact.get("field")) not in effect_keys]
        if event_facts:
            story_group.append((entity, event_facts))
        if other_facts:
            background.append((entity, other_facts))
    return ([story_group] if story_group else []) + list(_chunk(background, 2))


def _story_context_for_group(ledger, scenes, session, event_ids, event_by_id):
    """只传当前幕作用与上一幕 canonical events，不把自由摘要当事实喂回正文。"""
    if not ledger:
        return ""

    def _rank(scene):
        order = scene.get("order", 0)
        return scene.get("session", -1), order if isinstance(order, int) and not isinstance(order, bool) else 0

    event_ids = list(dict.fromkeys(x for x in event_ids if x))
    event_id_set = set(event_ids)
    current = [scene for scene in scenes
               if scene.get("session") == session
               and event_id_set.intersection(scene.get("event_refs") or [])]
    current.sort(key=lambda scene: (*_rank(scene), str(scene.get("scene_id") or "")))
    anchor = _rank(current[0]) if current else (session, -1)
    prior = [scene for scene in scenes
             if _rank(scene) < anchor]
    previous = max(prior, key=_rank, default=None)
    previous_events = []
    for event_ref in (previous.get("event_refs") or []) if previous else []:
        event = event_by_id.get(event_ref)
        if event:
            previous_events.append({key: event.get(key) for key in (
                "id", "label", "session", "participants", "effects")})
    payload = {
        "protagonist_ref": ledger.get("protagonist_ref"),
        "previous_scene": ({"scene_id": previous.get("scene_id"),
                            "canonical_events": previous_events}
                           if previous else None),
        "current_scenes": [
            {"scene_id": scene.get("scene_id"),
             "event_refs": [ref for ref in scene.get("event_refs", []) if ref in event_id_set],
             "dramatic_function": scene.get("dramatic_function")}
            for scene in current
        ],
    }
    context = ("\n【game Story Ledger（只用于组织本期行文）】"
               + json.dumps(payload, ensure_ascii=False)
               + "\n★previous_scene 只含已经发生的 canonical events；不得写出任何未来 scene。"
                 "事件参与者、结果、字段值仍只以本期 domain_events/facts 为准，"
                 "不得补造死亡、掉落、获得、阵营变化或其他动态真值。")
    return context


def _attach_story_provenance(docs, events, scenes, *, ws=None, session=None) -> None:
    """按单篇实际承载的事件写 provenance，避免把整组 refs 复制给每篇。"""
    event_to_scene = {
        event_ref: scene.get("scene_id")
        for scene in scenes
        for event_ref in (scene.get("event_refs") or [])
    }
    for doc in docs:
        content = str(doc.get("content") or "")
        if ws is not None:
            from pipeline.corpus_contract import reviewed_fidelity_provenance
            provenance = reviewed_fidelity_provenance(ws, session, doc)
            refs = provenance["event_refs"] if provenance is not None else []
        else:
            refs = [event.get("id") for event in events
                    if event.get("id") and _event_is_narrated(event, content)]
        doc["event_refs"] = refs
        doc["scene_refs"] = list(dict.fromkeys(
            event_to_scene[ref] for ref in refs if ref in event_to_scene))


class _CorpusReviewCallGuard:
    """Stop new corpus calls after a review execution failure in this render.

    Calls admitted before the failure may finish and retain their original trace.
    The lock only protects admission/failure state; provider calls stay parallel.
    """

    def __init__(self, tracer):
        self._tracer = tracer
        self.pfile = getattr(tracer, "pfile", None)
        self._lock = threading.Lock()
        self._failure = None

    def check(self):
        with self._lock:
            if self._failure is not None:
                raise self._failure

    def fail(self, error):
        with self._lock:
            if self._failure is None:
                self._failure = error
            failure = self._failure
        raise failure

    def chat_json(self, *args, **kwargs):
        self.check()
        return self._tracer.chat_json(*args, **kwargs)

    def chat_text(self, *args, **kwargs):
        self.check()
        return self._tracer.chat_text(*args, **kwargs)


def render_corpus(wp, ws, target_tokens, tracer, corpus, done_weeks, save_cb, log=print,
                  only_entities=None, only_entity_sessions=None, haystack_ratio=None):
    """only_entities=None:全量渲(每周全实体+filler)。
    only_entities=set:★增量 delta(§10.1)——【只渲这些新实体的 signal】并【追加】到已有周 docs,
    ``only_entity_sessions`` 精确补渲被新关系/事件改变的旧实体周；haystack_ratio
    显式设置时，所有正文验收后只追加尚缺的 filler。"""
    if target_tokens < 0:
        raise ValueError("语料规模不能为负数")
    corpus_scale(corpus, target_tokens, haystack_ratio)
    story_ledger = getattr(ws, "narrative", None) or {}
    quality_enabled = bool((wp.get("quality_contract") or {}).get("corpus_review"))
    from pipeline.corpus_contract import (canonical_context, review_documents, attach_receipts,
                                          authoritative_source_assertions, SOURCE_ASSERTION_INSTRUCTIONS,
                                          public_source_coverage_issues, CorpusReviewExecutionError,
                                          fidelity_requirements, fidelity_coverage_issues)
    projected = bool(getattr(ws, "disclosure", None))
    if (wp.get("quality_contract") or {}).get("public_disclosure") is True and not projected:
        raise ValueError("Public disclosure rendering was declared but its frozen plan is missing")
    if projected:
        from pipeline.disclosure import validate_plan, session_events as public_session_events
        issues = validate_plan(ws)
        if issues:
            raise ValueError(f"Public disclosure plan is invalid: {issues}")
        if not quality_enabled:
            raise ValueError("Public disclosure rendering requires the original semantic corpus review")
    # A review execution failure belongs to one independently rendered period.
    # Keep the existing guard inside that period so already-admitted sibling
    # calls can finish without dispatching follow-ups, but do not let one bad
    # period cancel every other period in the same corpus pass.
    base_tracer = tracer
    story_scenes = replay_story_ledger(ws, story_ledger) if story_ledger else []
    profile = wp.get("domain_profile", {})
    blueprint = getattr(ws, "world_blueprint", None) or wp.get("world_blueprint") or {}
    temporal = blueprint.get("temporal_model") or {}
    # blueprint 保存稳定机器枚举（chapter/week），正文、日志和 reviewer 必须共享
    # 同一个人类可读单位（章/周）；否则会生成“第5章”却授权“第5chapter”。
    time_unit = ws.period_unit()
    step_days = int(temporal.get("step_days", 7) or 7)
    sys_sig = _corpus_system(profile, blueprint, wp.get("style_spec"), semantic=quality_enabled)
    if projected:
        sys_sig += ("\n本任务按披露计划公开具体事实版本。文档日期是公开日期；"
                    "fact_session/fact_date 以及事件 session/date 是事实的原时点，两者不得混同。"
                    "依据各 disclosure_id 的 channel/acquisition_context 安排行文；这些载体说明不改变业务真值。"
                    "可以自然回顾旧事实或对比多个版本，但不得把旧值写成本期新发生的状态变化；"
                    "只使用本组公开目标和 CANON 已公开历史，不能补入其他私有事实。")
    sys_fil = _filler_system(profile, blueprint, wp.get("corpus_plan"))
    blocked = _tracked_blocklist(ws, profile)
    # 估算 filler/周 以达目标 token(~1字≈1token)。周并行后不再 early-stop;filler_per_week 已按目标分摊。
    n_sessions = ws.n_sessions
    filler_per_week = _filler_documents_per_session(wp, target_tokens, n_sessions)
    by_id = {x["session_id"]: x for x in corpus["sessions"]}
    delta_mode = only_entities is not None or only_entity_sessions is not None
    only_entities = set(only_entities or [])
    only_entity_sessions = set(only_entity_sessions or [])
    weeks = list(ws.sessions()) if delta_mode else [s for s in ws.sessions() if s not in done_weeks]
    lock = threading.Lock()
    fallback_count: list[int] = []                        # 耗尽兜底计数(list.append 线程安全;验收要求趋零)

    def _render_week(s):                                  # ★一周的全部渲染 = 一个并行单元
        review_guard = _CorpusReviewCallGuard(base_tracer)
        tracer = review_guard
        date = _date_of(s, step_days=step_days)
        review_context = canonical_context(ws, s)
        facts = _session_facts(ws, s)
        event_decls = {e.get("id"): e for e in blueprint.get("event_types", [])}
        event_candidates = (public_session_events(ws, s) if projected else
                            (getattr(ws, "events", None) or []))
        labeled_events = [
            {**event, "label": (event_decls.get(event.get("type")) or {}).get(
                "label", event.get("type", ""))}
            for event in event_candidates
        ]
        event_by_id = {event.get("id"): event for event in labeled_events if event.get("id")}
        session_events = (labeled_events if projected else
                          [event for event in labeled_events if event.get("session") == s])
        if delta_mode:                                    # ★delta:新实体全程 + 旧实体受结构变化的精确 session
            facts = [f for f in facts if (f["entity"] in only_entities
                                          or (f["entity"], s) in only_entity_sessions)]
            if projected:
                session_events = [event for event in session_events if any(
                    name in only_entities or (name, s) in only_entity_sessions
                    for name in (event.get("participants") or {}).values())]
            if not facts and not (projected and session_events):
                return s                                  # 新实体本周无事实 → 不加 doc
        bysku = {}
        for f in facts:
            bysku.setdefault(f["entity"], []).append(f)
        if projected:
            # Publication occurrences, rather than truth-change periods, are
            # the authoring units. An event-only publication still needs a group.
            occurrence_ids = list(dict.fromkeys(item["disclosure_id"] for item in facts + session_events))
            sig_groups = []
            for occurrence in occurrence_ids:
                group_facts = [item for item in facts if item["disclosure_id"] == occurrence]
                group_events = [item for item in session_events if item["disclosure_id"] == occurrence]
                names = list(dict.fromkeys([item["entity"] for item in group_facts]
                    + [name for event in group_events for name in (event.get("participants") or {}).values()]))
                sig_groups.append(([(name, [item for item in group_facts if item["entity"] == name])
                                    for name in names], group_events))
        elif story_ledger:
            sig_groups = [(group, None) for group in _story_signal_groups(bysku, session_events)]
        else:
            sig_groups = [(group, None) for group in _chunk(list(bysku.items()), 2)]
        n_batches = filler_per_week

        def _render_sig(group):                           # ★信号块:渲全 + 渲对
            grp, planned_events = group
            from pipeline.grounding import STOP_MARKERS    # ★只借停用标记(STOP_MARKERS);忠实检不再用 §G 的 attributed(死钉②不同尺)
            gf = [f for _e, fs in grp for f in fs]
            group_entities = {name for name, _facts in grp}
            group_context = {**review_context, "source_assertions": (
                authoritative_source_assertions(ws, s, {(f["entity"], f["field"]) for f in gf})
                if quality_enabled else [])}
            if planned_events is not None:
                group_events = planned_events
            elif story_ledger:
                group_fact_keys = {(f.get("entity"), f.get("field")) for f in gf}
                group_events = [
                    event for event in session_events
                    if any((effect.get("entity"), effect.get("field")) in group_fact_keys
                           for effect in (event.get("effects") or []) if isinstance(effect, dict))
                ]
            else:
                group_events = [e for e in session_events
                                if group_entities.intersection((e.get("participants") or {}).values())]
            # Private scene functions/history must not reintroduce unrevealed
            # outcomes. Planned runs already carry public history in CANON.
            story_context = ("" if projected else _story_context_for_group(
                story_ledger, story_scenes, s, [e.get("id") for e in group_events], event_by_id))
            requirements = (fidelity_requirements(ws, s, facts=gf, events=group_events)
                            if quality_enabled else [])
            # 待渲事实:非停用 → 派盲判别器读 (实体,字段) 的值,代码量纲严格对账;停用 → 验 (实体,停用标记) 同篇
            want_val = [(f["entity"], f["field"], str(f["value"])) for f in gf if f.get("value") and not f.get("stopped")]
            want_stop = [(f["entity"], f["field"]) for f in gf if f.get("stopped")]

            def _discriminate(contents):
                """★盲判别器忠实检(死钉①③):对每个 want_val atom 派一个盲读者只读 contents 答值,
                代码量纲严格对账;对每个 want_stop atom 验停用标记同篇。
                返回 (miss_val, miss_stop, miss_event)，供 missing/hint 管道复用。"""
                # 同一组文档只让盲读者读一次；true_value 仍只在代码对账侧，绝不进 prompt。
                queries = [
                    {"key": f"q{index}", "entity": entity, "field": field}
                    for index, (entity, field, _value) in enumerate(want_val)
                ]
                answers = _discriminate_many(contents, queries, tracer)
                miss_val = []
                for index, (entity, field, value) in enumerate(want_val):
                    answer = answers.get(f"q{index}", "")
                    if _strict_eq(answer, str(value)):
                        continue
                    read = "不确定" if _is_unsure(answer) else answer
                    miss_val.append(
                        f"{entity}的「{field}」(应承载值={value}):"
                        f"盲读者据文档读出的是『{read}』,与应承载的值不一致/读不出")
                # stopped atom:期望判别器读出"停止/不再统计"语义;此处复用 §G 停用标记同篇检(refusal 语义,非 _strict_eq)
                miss_stop = [f"{e}的「{fl}」应让读者读出『自本期停止统计』,但文档未表达停用"
                             for (e, fl) in want_stop
                             if not any((e in c) and any(m in c for m in STOP_MARKERS) for c in contents)]
                miss_event = _missing_event_narratives(group_events, contents)
                return miss_val, miss_stop, miss_event
            # ★禁词豁免(014559 尸检:词表「累计」撞字段名「累计计费工时」→ 整篇核验前被静默丢,27/27 弃题同根)。
            #   豁免集 = 本组【所有被要求逐字出现的串】= 字段名+实体名+事实值(刀1审计:值含禁词如「维持治疗」
            #   时,'逐字照抄'与'禁全局口径词'否则构成不可满足约束 → 4 轮必废 → 兜底吸收症状)。
            #   单一真源(由本组事实派生,非按域手维护);长串先遮,防短串是长串子串。
            exempt = sorted({f["field"] for f in gf} | {f["entity"] for f in gf}
                            | {str(f["value"]) for f in gf if f.get("value")}
                            | {item["source"] for item in group_context["source_assertions"]},
                            key=len, reverse=True)

            def _leaks(text):
                masked = text
                for nm in exempt:
                    masked = masked.replace(nm, "■" * len(nm))
                return [b for b in LEAK_BANNED if b in masked]

            grp_docs, hint, left = [], "", []
            for _att in range(4):                         # 多给几次重渲机会,强制渲全(世界辛苦生成,必须全用上)
                out = tracer.chat_json("render.signal",
                    [{"role": "system", "content": sys_sig},
                     {"role": "user", "content": render(
                         "corpus.quality_user" if quality_enabled else "corpus.user",
                         s=week_label(s), time_unit=time_unit, date=date,
                         facts=json.dumps(gf, ensure_ascii=False),
                         events=json.dumps(group_events, ensure_ascii=False),
                         story_context=story_context, hint=hint)
                         + (("\n【生成与审阅共享的截至时点事实；不得把未知业务状态写成已发生】\n"
                             + json.dumps(group_context, ensure_ascii=False)
                             + ("\n" + SOURCE_ASSERTION_INSTRUCTIONS
                                if group_context["source_assertions"] else "")) if quality_enabled else "")}],
                    temperature=0.6 if _att == 0 else 0.2, max_tokens=SIGNAL_MAX_TOKENS,
                    **({"response_format": {"type": "json_object"}} if quality_enabled else {}))
                if quality_enabled and (not isinstance(out, dict) or "__error__" in out
                        or not isinstance(out.get("docs"), list) or not out["docs"]
                        or any(not isinstance(doc, dict) or not isinstance(doc.get("content"), str)
                               or not doc["content"].strip() for doc in out["docs"])):
                    review_guard.fail(RuntimeError("render.signal failed or returned invalid document shape"))
                cand, leak_notes = [], []
                for d in _dicts(out.get("docs") if isinstance(out, dict) else []):
                    if not d.get("content"):
                        continue
                    hits = _leaks(d["content"])
                    if hits and not quality_enabled:       # legacy 字面拒绝；quality 仅把提示送语义审阅
                        leak_notes.append(f"《{(d.get('title') or d.get('type') or '无题')}》因使用全局口径词{hits}被废弃")
                    else:
                        # signal/filler 身份由代码决定，模型不能用额外元数据让已验收
                        # 的事件文档在 sanitize 阶段被当草堆删除。
                        clean = {key: d[key] for key in ("title", "type", "content", "fact_refs")
                                 if key in d}
                        clean["is_filler"] = False
                        cand.append(clean)
                contents = [d.get("content", "") for d in cand]
                blind_reads, lexical_diagnostics = [], []
                if quality_enabled:
                    # Queries contain neither expected values nor stop/event targets.
                    queries = [{"key": f"q{index}", **{key: fact[key] for key in (
                                    "entity", "field", "fact_session", "fact_date", "disclosure_id", "target_ref")
                                    if key in fact}}
                               for index, fact in enumerate(gf)]
                    try:
                        answers = _discriminate_many(contents, queries, tracer, semantic=True)
                    except Exception as exc:
                        review_guard.fail(exc)
                    blind_reads = [{**query, "answer": answers[query["key"]]} for query in queries]
                    lexical_diagnostics = [
                        {"doc_index": index, "global_scope_word_hits": _leaks(doc.get("content", ""))}
                        for index, doc in enumerate(cand) if _leaks(doc.get("content", ""))]
                    lexical_diagnostics += [
                        {"kind": "literal_value_mismatch", "entity": fact["entity"], "field": fact["field"],
                         "blind_answer": answers[f"q{index}"]}
                        for index, fact in enumerate(gf) if not fact.get("stopped")
                        and not _strict_eq(answers[f"q{index}"], str(fact["value"]))]
                    lexical_diagnostics += [{"kind": "event_literal_hint", "hint": hint}
                                            for hint in _missing_event_narratives(group_events, contents)]
                    lexical_diagnostics += [
                        {"kind": "stop_literal_hint", "entity": fact["entity"], "field": fact["field"]}
                        for fact in gf if fact.get("stopped")
                        and not any(fact["entity"] in content and any(word in content for word in STOP_MARKERS)
                                    for content in contents)]
                    miss_val, miss_stop, miss_event = [], [], []
                else:
                    miss_val, miss_stop, miss_event = _discriminate(contents)
                missing = miss_val + miss_stop + miss_event
                unsupported = []
                quality_review = None
                if not missing and quality_enabled:
                    quality_review = review_documents(tracer, ws, s, cand, context=group_context,
                        requirements=requirements, blind_reads=blind_reads,
                        lexical_diagnostics=lexical_diagnostics)
                    if quality_review["status"] == "error":
                        # The real tracer has already preserved the attempted call/raw
                        # output. Do not feed a timeout or malformed review to the
                        # author as an unsupported business assertion.
                        review_guard.fail(CorpusReviewExecutionError(quality_review, s))
                    if quality_review["status"] != "passed":
                        unsupported = [json.dumps(item, ensure_ascii=False) for item in quality_review["issues"]]
                elif not missing and story_ledger:
                    unsupported = review_narrative_supportedness(
                        tracer,
                        canon={"session": s, "facts": gf, "domain_events": group_events,
                               "period_label": f"第{week_label(s)}{time_unit}",
                               "document_date": date, "time_unit": time_unit,
                               "allowed_document_scaffolding": ["日志记录", "档案登记", "通报提及"],
                               "allowed_past_context": story_context},
                        candidate=cand,
                        scope=f"game corpus session {s}")
                left = missing + [f"语义审阅缺口:{item}" for item in unsupported]
                grp_docs = cand
                if not left:
                    break
                # ★hint 如实(老版把"写了但犯禁被废"误报成"没写"→ 重试不收敛):缺什么、为什么缺,分开说
                hint = ""
                if missing:
                    hint += (f"\n★ 这些事实在上一版【没有合格呈现】"
                             f"(实体名与值必须就近、值逐字照抄):{missing}。")
                if miss_event:
                    exact_effects = [
                        {"entity": effect.get("entity"), "field": effect.get("field"),
                         "set": effect.get("set", effect.get("value"))}
                        for event in group_events for effect in (event.get("effects") or [])
                        if isinstance(effect, dict)
                    ]
                    hint += ("\n★ 只修上述事件证据：同一篇中明确写动作，并逐字写出这些"
                             f" entity/field/set，禁止同义替换：{json.dumps(exact_effects, ensure_ascii=False)}。")
                if unsupported:
                    hint += (f"\n★ 上一版语义审阅发现:{unsupported}。"
                             "按给定 facts/events、当期来源声明与已发生上下文修正，"
                             "补齐缺失的来源对应关系，删除无依据断言，不创造新事实。")
                if leak_notes:
                    hint += (f"\n★ 另:上一版 {leak_notes}——重写时把其中事实写进正文,但【删掉这些全局口径词】"
                             f"(注意:字段名/实体名/事实值本身含这些字的照常写,不算犯禁)。")
                hint += ("按语义意见修订正文，准确区分本期、历史与未发生内容。" if quality_enabled
                         else "逐条重写进正文(仍只写本期)。")
            else:
                fallback_count.append(len(left))          # ★机械验收落点(语义改为"弃段计数"):汇总进末尾日志
                ents = sorted({m.split("的「")[0] for m in left})
                log(f"  ⚠fail-loud弃段[{time_unit}{week_label(s)}]:{len(left)} 个 atom 多轮重渲后盲读者仍不可还原,弃段不入库({ents})")
                if quality_enabled or (story_ledger and group_events):
                    raise RuntimeError(
                        f"正文质量审阅未通过:{time_unit}{week_label(s)} 经 4 轮生成与审阅仍有 {len(left)} 个未通过项")
                return []
            if quality_enabled:
                attach_receipts(grp_docs, quality_review, s)
            if story_ledger:
                _attach_story_provenance(grp_docs, group_events, story_scenes,
                    **({"ws": ws, "session": s} if quality_enabled else {}))
            return grp_docs

        filler_failures: list[str] = []

        def _render_fil(_ci):                             # 一次只生成一篇纯正文
            out = tracer.chat_text("render.filler",
                [{"role": "system", "content": sys_fil},
                 {"role": "user", "content": render("filler.user", s=week_label(s), time_unit=time_unit,
                                                      date=date, slot=_ci + 1)}],
                temperature=0.9, max_tokens=FILLER_TEXT_MAX_TOKENS)
            try:
                return _accept_filler_text(out, blocked)
            except RuntimeError as error:
                # filler 只提供草堆密度，不承载任何 gold。单篇协议失败显式记账并跳过；
                # 最终仍以总语料字符下限 fail-closed，不重试也不伪造替代正文。
                filler_failures.append(str(error))
                return []

        def render_signal_with_checkpoint(group):
            from pathlib import Path
            from pipeline.semantic_review import fingerprint
            from pipeline.run import _atomic_write_json
            from pipeline import corpus_contract
            pfile = getattr(tracer, "pfile", None)
            cache_path = None
            if isinstance(pfile, Path):
                key = fingerprint({"wp": wp, "world": ws.to_dict(), "session": s, "group": group,
                    "model": config.MODEL, "reviewer": config.REVIEWER_MODEL,
                    "discriminator": config.DISCRIMINATOR_MODEL,
                    "render": Path(__file__).read_text(encoding="utf-8"),
                    "review": Path(corpus_contract.__file__).read_text(encoding="utf-8")})
                cache_path = pfile.parent / "05_signal_checkpoints" / (key + ".json")
                if cache_path.exists():
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    if cached.get("key") != key or cached.get("hash") != fingerprint(cached.get("documents")):
                        raise ValueError("Corpus signal checkpoint changed")
                    return cached["documents"]
            documents = _render_sig(group)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_json(cache_path, {"key": key, "documents": documents, "hash": fingerprint(documents)})
            return documents

        sig_lists = config.pmap(render_signal_with_checkpoint, sig_groups, workers=4)
        review_guard.check()
        # Never hold the corpus commit lock across network calls. Four periods
        # may render concurrently; two filler workers per period bound fan-out.
        fil_lists = config.pmap(_render_fil, list(range(n_batches)), workers=2) if not delta_mode else []
        conflict_docs = _render_conflict_docs(ws, s, date, tracer) if not delta_mode else []
        with lock:                                         # 周乱序完成 → 锁内更新+逐周存盘(断点续渲不丢)
            if delta_mode:                                 # ★delta:追加结构变化 signal,接着编号;不灌 filler/conflict
                docs = list(by_id.get(s, {}).get("docs", []))
                base = len(docs)
                for gl in sig_lists:
                    for d in gl:
                        d["doc_id"] = f"s{s}_sig_{base}"; base += 1; docs.append(d)
                by_id[s] = {"session_id": s, "date": date, "docs": docs}   # 旧 docs 原样保留,只增量
            else:                                          # 全量:本周 signal + filler + 小道矛盾,整周写入
                if filler_failures:
                    log(f"  ⚠{time_unit}{week_label(s)} filler 缺失 {len(filler_failures)}/{n_batches}:"
                        f"{filler_failures[:2]}")
                docs = []                                  # 周内顺序编号,避免 race(doc_id 含 s,跨周不撞)
                for gl in sig_lists:
                    for d in gl:
                        d["doc_id"] = f"s{s}_sig_{len(docs)}"; docs.append(d)
                for fl in fil_lists:
                    for d in fl:
                        d.update({"doc_id": f"s{s}_fil_{len(docs)}", "is_filler": True, "fact_refs": []}); docs.append(d)
                for d in conflict_docs:                                # ★L5:本周小道矛盾文档(非 L5 场景为空)
                    d.update({"doc_id": f"s{s}_conf_{len(docs)}", "is_conflict": True, "fact_refs": []}); docs.append(d)
                for d in _render_sensitive_docs(ws, s, date):             # ★L10:本周敏感写入文档(确定性模板,X 逐字就近;非 L10 场景为空)
                    d.update({"doc_id": f"s{s}_sens_{len(docs)}", "is_sensitive": True, "fact_refs": []}); docs.append(d)
                for d in _render_rule_docs(ws, s, date):                  # ★L9:本周条件归纳执行实例(确定性单条情境→动作;非 L9 场景为空)
                    d.update({"doc_id": f"s{s}_rule_{len(docs)}", "is_rule_instance": True, "fact_refs": []}); docs.append(d)
                by_id[s] = {"session_id": s, "date": date, "docs": docs}
                done_weeks.add(s)
            corpus["sessions"] = [by_id[k] for k in sorted(by_id)]
            save_cb()
            ch = sum(len(dd.get("content", "")) for x in corpus["sessions"] for dd in x["docs"])
            tag = "delta+" if delta_mode else ""
            log(f"  [{time_unit} {week_label(s)} ✓{tag} {len(done_weeks)}/{n_sessions}] 累计 {sum(len(x['docs']) for x in corpus['sessions'])} 篇 / {ch/1e6:.2f}M 字")
        return s

    period_failures = []
    period_failure_lock = threading.Lock()

    def _render_week_isolated(s):
        try:
            return _render_week(s)
        except CorpusReviewExecutionError as exc:
            # The exact failed review is already present in the tracer.  Keep
            # the period absent from done_weeks, allow the remaining periods to
            # finish and checkpoint, then fail the stage once with the original
            # typed error so the normal partial-corpus recovery path is used.
            with period_failure_lock:
                period_failures.append((s, exc))
            log(f"  ⚠{time_unit}{week_label(s)}语料审阅执行失败；本期保持待补，其他期继续")
            return None

    config.pmap(_render_week_isolated, weeks, workers=max(1, min(4, len(weeks))))  # Bound queued week/group memory too.
    if period_failures:
        period_failures.sort(key=lambda item: item[0])
        raise period_failures[0][1]
    if _render_public_stage_material(wp, ws, tracer, corpus):
        save_cb()
    sanitized = _sanitize_corpus(corpus, ws, profile, semantic=quality_enabled)
    from pipeline.corpus_contract import public_rule_coverage_issues
    rule_issues = public_rule_coverage_issues(ws, corpus)
    if rule_issues:
        raise RuntimeError(f"Public stage material missing after corpus sanitization: {rule_issues}")
    if quality_enabled:
        source_issues = public_source_coverage_issues(ws, corpus) + fidelity_coverage_issues(ws, corpus)
        if source_issues:
            raise RuntimeError("Public source material needs a complete new group review: "
                               + json.dumps(source_issues, ensure_ascii=False))
    if story_ledger and not projected:
        covered = {
            event_ref
            for session in corpus.get("sessions", [])
            for doc in session.get("docs", [])
            for event_ref in (doc.get("event_refs") or [])
        }
        expected = {str(event.get("id")) for event in (getattr(ws, "events", None) or [])
                    if isinstance(event, dict) and event.get("id")}
        missing_events = sorted(expected - covered)
        if missing_events:
            # 清掉缺证据事件所在周的 checkpoint，人工续跑时会真正重渲，而非
            # 反复读取同一份坏断点。
            missing_sessions = {
                event.get("session") for event in (getattr(ws, "events", None) or [])
                if isinstance(event, dict) and str(event.get("id")) in missing_events
            }
            done_weeks.difference_update(missing_sessions)
            corpus["sessions"] = [session for session in corpus.get("sessions", [])
                                  if session.get("session_id") not in missing_sessions]
            save_cb()
            raise RuntimeError(f"game narrative 缺 canonical event 正文证据:{missing_events}")
    if any(sanitized.values()):
        save_cb()
        log(f"  ✓ 语料收口:规范化 fact_refs {sanitized['canonicalized_refs']} 篇 / "
            f"弃无引用信号 {sanitized['dropped_unref']} 篇 / "
            f"清理扩容后撞词 filler {sanitized['dropped_filler_leaks']} 篇 / "
            f"清理凭据形态 filler {sanitized['dropped_filler_credentials']} 篇")
    if haystack_ratio is not None:
        _top_up_haystack(corpus, target_tokens, haystack_ratio, base_tracer,
                        sys_fil, blocked, time_unit, save_cb, log)
    ch = sum(len(dd.get("content", "")) for x in corpus["sessions"] for dd in x["docs"])
    if ch < target_tokens and haystack_ratio is None:
        save_cb()
        raise RuntimeError(
            f"语料字符不足:{ch}/{target_tokens}；filler 可单篇缺失，但总规模合同不允许欠账")
    fb = f";⚠fail-loud弃段 {len(fallback_count)} 处/{sum(fallback_count)} 个 atom 未忠实渲染(验收要求趋零)" if fallback_count else ";弃段 0(✓)"
    log(f"  ✓ 渲染完成:{sum(len(x['docs']) for x in corpus['sessions'])} 篇 / {ch/1e6:.2f}M 字(目标 {target_tokens/1e6:.1f}M){fb}")


PHRASE_SYS = render("phrase.system")


def phrase_questions(orders, wp, tracer, log=print, *, audit=None, checkpoint_path=None) -> list[dict]:
    from copy import deepcopy
    from pipeline.question_contract import attach_question_contract, validate_question

    if audit is not None and not isinstance(audit, dict):
        raise TypeError("Question wording audit must be a dictionary")
    orders = list(orders)
    # Production uses an item checkpoint. The compatibility path without a
    # checkpoint retains its historical fail-fast call contract.
    from pathlib import Path
    from pipeline.semantic_review import fingerprint
    from pipeline import question_wording
    from pipeline.run import _atomic_write_json
    persistent = checkpoint_path is not None
    checkpoint = Path(checkpoint_path) if persistent else None
    saved_items, progress_lock = {}, threading.Lock()
    execution_binding = fingerprint({"whitepaper": wp, "author": config.MODEL,
        "reviewer": config.REVIEWER_MODEL, "render": Path(__file__).read_text(encoding="utf-8"),
        "wording": Path(question_wording.__file__).read_text(encoding="utf-8")})
    if checkpoint and checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("binding") == execution_binding:
            if saved.get("hash") != fingerprint({k:v for k,v in saved.items() if k != "hash"}):
                raise ValueError("Question progress checkpoint changed")
            saved_items = saved["items"]
    def remember(order, row, result):
        if checkpoint:
            with progress_lock:
                saved_items[fingerprint(order)] = {"row": deepcopy(row), "result": deepcopy(result)}
                value = {"binding": execution_binding, "items": deepcopy(saved_items)}
                value["hash"] = fingerprint(value)
                _atomic_write_json(checkpoint, value)
    report = audit if audit is not None else {}
    report.clear()
    report.update(version="original-question-wording-batch/v2", execution_status="running",
                  original_count=len(orders), returned_qids=[], counts={},
                  items=[{"input_index": index, "source_qid": order.get("qid"),
                          "qid": order.get("qid"), "original_order": deepcopy(order),
                          "status": "not_started", "calls_admitted": 0, "attempts": [],
                          "candidate": None, "selected": False, "failure": None}
                         for index, order in enumerate(orders)])
    # Per-run admission state. Do not serialize provider calls; already-admitted
    # calls can finish and keep their normal tracer records after a peer fails.
    failure_lock, failures = threading.Lock(), []
    semantic_mode = bool((wp.get("quality_contract") or {}).get("scoring_policy"))

    def _phrase_one(o, row):                              # 每条订单独立 → 并发出题
        def call_json(*args, **kwargs):
            if persistent:
                kwargs["retries"] = 3
            with failure_lock:
                if semantic_mode and failures and not persistent:
                    raise RuntimeError("Previous question wording execution failed; no new calls")
                row["calls_admitted"] += 1
            try:
                output = tracer.chat_json(*args, **kwargs)
            except Exception as exc:
                with failure_lock:
                    if semantic_mode and not failures and not persistent:
                        failures.append(exc)
                raise
            if semantic_mode and isinstance(output, dict) and "__error__" in output:
                with failure_lock:
                    if not failures and not persistent:
                        failures.append(RuntimeError("Question wording provider returned an execution error"))
            return output

        o = attach_question_contract(o, wp)
        row["qid"] = o.get("qid")
        line = line_for(o.get("line", ""))                # 出题意图/须隐藏 = 各产线自己的 intent()
        if line is None:                                  # 兜底(订单都来自已建线,理论不触发)
            return {**o, "question": "", "_phrase_fallback": False}
        intent, hide = line.intent(o)
        if o["question_contract"]["render_policy"] == "semantic_review":
            from pipeline.question_wording import AUTHOR_SYSTEM, review_wording
            if getattr(line, "deterministic_phrasing", False):
                text = intent
            else:
                authored = call_json("phrase", [{"role": "system", "content": AUTHOR_SYSTEM},
                    {"role": "user", "content": render("phrase.user", intent=intent, hide=hide)}],
                    temperature=0.5, max_tokens=2048, retries=1, strict_json=True)
                row["author_output"] = deepcopy(authored)
                if (not isinstance(authored, dict) or "__error__" in authored
                        or not isinstance(authored.get("question"), str) or not authored["question"].strip()):
                    raise RuntimeError("Question author failed; no semantic fallback on execution failure")
                text = authored["question"]
            row["candidate"] = {**deepcopy(o), "question": text}
            row["attempts"].append({"source": "canonical" if getattr(line, "deterministic_phrasing", False)
                                    else "author", "question": text, "review": None})
            first = review_wording(text, o["question_contract"], chat_json=call_json,
                                   model=config.REVIEWER_MODEL)
            row["attempts"][-1]["review"] = deepcopy(first)
            reviews = [first]
            if first["status"] != "passed" and not getattr(line, "deterministic_phrasing", False):
                # Return semantic feedback to the same author once. Execution
                # failures propagate without inventing a repaired candidate.
                repair_request = {"original_intent": intent, "hidden_values": hide,
                                  "candidate_question": text, "review_feedback": first["opinion"]}
                row["repair_request"] = deepcopy(repair_request)
                repaired = call_json("phrase", [{"role": "system", "content": AUTHOR_SYSTEM},
                    {"role": "user", "content": json.dumps(repair_request, ensure_ascii=False)}],
                    temperature=0.5, max_tokens=2048, retries=1, strict_json=True)
                row["repair_author_output"] = deepcopy(repaired)
                if (not isinstance(repaired, dict) or "__error__" in repaired
                        or not isinstance(repaired.get("question"), str) or not repaired["question"].strip()):
                    raise RuntimeError("Question repair author failed; no semantic fallback on execution failure")
                text = repaired["question"]
                row["candidate"] = {**deepcopy(o), "question": text}
                row["attempts"].append({"source": "author_repair", "question": text, "review": None})
                reviews.append(review_wording(text, o["question_contract"],
                    chat_json=call_json, model=config.REVIEWER_MODEL))
                row["attempts"][-1]["review"] = deepcopy(reviews[-1])
            final = reviews[-1]
            row["status"] = "passed" if final["status"] == "passed" else "unresolved"
            return {**o, "question": text if final["status"] == "passed" else "",
                    "_phrase_fallback": len(reviews) > 1,
                    "question_validation": {"status": final["status"], "mode": "semantic_review",
                        "source": "llm_meaning_review", "semantic_review": final,
                        "review_history": reviews}}
        fallback_issues = validate_question(intent, o["question_contract"])
        if fallback_issues:
            return {**o, "question": "", "_phrase_fallback": False,
                    "question_validation": {"status": "rejected", "issues": fallback_issues}}
        # ★确定性出题 bypass(L9 闭选项 MC):选项串必须逐字保真、LLM 润色会打乱选项/丢 gold → 破坏纯代码 EM。
        #   直接用 intent 原文作题面(它已是完整可答的 MC 题,含 held-out x* + 全部选项)。
        if o["question_contract"]["render_policy"] == "canonical_template":
            return {**o, "question": intent, "_phrase_fallback": False,
                    "question_validation": {"status": "passed", "mode": "canonical_template",
                        "source": "deterministic_template", "llm_calls": 0,
                        "issues": [], "rewrite_issues": []}}
        out = call_json("phrase",
            [{"role": "system", "content": PHRASE_SYS},
             {"role": "user", "content": render("phrase.user", intent=intent, hide=hide)}],
            temperature=0.5, max_tokens=2048)
        q = out.get("question", "") if isinstance(out, dict) else ""
        q = q if isinstance(q, str) else ""
        issues = validate_question(q, o["question_contract"])
        fell_back = bool(issues)
        if fell_back:
            # intent 是产线代码生成、已被良定义闸验证过的完整可答题面；润色模型只负责
            # 表达，不拥有订单生杀权。协议失败时直接保留真源，不丢掉已验证的供给。
            q = intent
        # Validate the fallback through the same contract as model output.
        final_issues = validate_question(q, o["question_contract"])
        if final_issues:
            q = ""
        return {**o, "question": q, "_phrase_fallback": fell_back,
                "question_validation": {"status": "rejected" if final_issues else "passed",
                    "mode": "template_fallback" if fell_back else o["question_contract"]["render_policy"],
                    "source": "contract_validator", "llm_calls": 1,
                    "issues": final_issues, "rewrite_issues": issues}}
    def _ph(indexed):
        index, order = indexed
        row = report["items"][index]
        previous = deepcopy(saved_items.get(fingerprint(order))) if persistent else None
        if previous and previous.get("result") is not None:
            result = previous["result"]
            if (not result.get("question") or not semantic_mode or not question_wording.validate_wording(result)):
                row.update(previous["row"], input_index=index, resumed=True, calls_admitted=0)
                return deepcopy(result)
        if previous and previous["row"].get("failure"):
            row["previous_execution_failure"] = previous["row"]["failure"]
        try:
            result = _phrase_one(order, row)
        except Exception as exc:
            with failure_lock:
                previously_failed = bool(failures)
                if semantic_mode and not failures and not persistent:
                    failures.append(exc)
            row["status"] = ("not_run_after_execution_failure"
                             if not persistent and previously_failed and not row["calls_admitted"] else "execution_error")
            row["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(getattr(exc, "report", None), dict):
                row["failed_review"] = deepcopy(exc.report)
                if row["attempts"] and row["attempts"][-1]["review"] is None:
                    row["attempts"][-1]["review"] = deepcopy(exc.report)
            remember(order, row, None)
            if persistent:
                return {**order, "question": "", "_phrase_fallback": False}
            raise
        row["selected"] = bool(result.get("question", "").strip())
        if row["status"] == "not_started":
            row["status"] = "passed" if row["selected"] else "rejected"
        # Keep the last actual candidate text even when the returned selection
        # intentionally has an empty question to exclude unresolved wording.
        actual_text = (row["candidate"] or {}).get("question", result.get("question", ""))
        row["candidate"] = {**deepcopy(result), "question": actual_text}
        row["candidate"].pop("_phrase_fallback", None)
        remember(order, row, result)
        return result

    def update_counts():
        from collections import Counter
        report["counts"] = dict(Counter(row["status"] for row in report["items"]))

    try:
        raw = config.pmap(_ph, enumerate(orders), workers=8)
    except Exception as exc:
        report["execution_status"] = "failed"
        report["execution_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        update_counts()
        raise
    fallback_count = sum(bool(q.pop("_phrase_fallback", False)) for q in raw)
    qs = [q for q in raw if q.get("question", "").strip()]    # 丢并发下偶发的空题面
    dropped = len(raw) - len(qs)
    errors = [row for row in report["items"] if row["status"] == "execution_error"]
    report["execution_status"] = "completed_with_errors" if errors else "completed"
    report["returned_qids"] = [q.get("qid") for q in qs]
    update_counts()
    if errors and not qs:
        report["execution_status"] = "failed"
        raise RuntimeError("No question wording completed; per-item progress retained")
    by_line = {}
    for q in qs:
        by_line[q.get("line", "?")] = by_line.get(q.get("line", "?"), 0) + 1
    log(f"  ④ 出题:{len(qs)} 题完成(返修或兜底 {fallback_count};丢空 {dropped};桥实体/答案不进题面);by_line {by_line}")
    return qs


# ════════════════════════════════════════════════════════════════════════════
# 单测:盲判别器忠实检(★不打真 API——monkeypatch tracer.chat_json 返回桩)
#   跑法:./venv/bin/python pipeline/render.py
# ════════════════════════════════════════════════════════════════════════════
def _self_test() -> bool:
    checks = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    # ── 量纲严格对账原语:保留 %,绝不归一蒙混 ──
    ck("_strict_eq 同量纲相等", _strict_eq("0.78", "0.78"))
    ck("★双量纲不等(78% != 0.78)", not _strict_eq("78%", "0.78"))
    ck("★裸数不等(78 != 0.78)", not _strict_eq("78", "0.78"))
    ck("★带%与不带不等(78% != 78)", not _strict_eq("78%", "78"))
    ck("全角→半角后相等", _strict_eq("０.７８", "0.78"))
    ck("含单位逐字相等", _strict_eq("320万", "320万"))
    ck("'不确定'判不还原", not _strict_eq("不确定", "0.78"))
    ck("空答判不还原", not _strict_eq("", "0.78"))

    # ── _discriminator_recovers 端到端(桩判别器:据 docs 抠出 entity 那句里的值;★绝不看 true_value)──
    #   桩模拟盲读者:从 docs 里找含 entity 的句子、读出"为/是"后面那个 token;读不出/歧义回"不确定"。
    def _fake_tracer_factory():
        class _T:
            seen_prompts = []   # 留痕:验判别器 prompt 里【绝不含】true_value(死钉③)

            def chat_json(self, step, messages, **kw):
                user = messages[1]["content"]
                _T.seen_prompts.append(user)
                # 桩盲读者:从 user 里夹的 docs 文本读 entity 的值(纯字符串启发,不依赖外部世界)
                docs = user
                import re as _re
                # 找 "<...命中率...>0.78" 或 "命中率78%" 这类:简单抓"命中率"后第一个数字串(含%)
                if "命中率" in docs:
                    m = _re.search(r"命中率[为是:]?\s*([0-9.]+%?)", docs)
                    if m:
                        return {"answers": [{"key": "q0", "answer": m.group(1)}]}
                # 负责人歧义:文档里出现两个不同负责人 → 盲读者答"不确定"
                if "负责人" in docs:
                    names = set(_re.findall(r"负责人[为是:]?\s*([一-龥]{2,3})", docs))
                    if len(names) == 1:
                        return {"answers": [{"key": "q0", "answer": names.pop()}]}
                    return {"answers": [{"key": "q0", "answer": "不确定"}]}    # 多负责人歧义 / 读不出
                # K=V 句 "X本期「Y」为Z":抠 Z
                m = _re.search(r"为([0-9.]+%?)", docs)
                if m:
                    return {"answers": [{"key": "q0", "answer": m.group(1)}]}
                return {"answers": [{"key": "q0", "answer": "不确定"}]}
        return _T()

    # (a) 干净自然句:命中率0.78 → 判别器还原 0.78 → pass
    t = _fake_tracer_factory()
    docs_a = ["本周天枢数据部运行平稳,经统计本期命中率0.78,团队士气高涨。"]
    ok_a, ans_a = _discriminator_recovers(docs_a, "天枢数据部", "命中率", "0.78", t)
    ck("(a) 干净自然句 0.78 → 还原 pass", ok_a and ans_a.startswith("0.78"))

    # (b) 双量纲:文档写 78%、世界 0.78 → 判别器读出 78% ≠ 0.78 → fail
    t = _fake_tracer_factory()
    docs_b = ["本周天枢数据部表现优异,本期命中率78%,继续保持。"]
    ok_b, ans_b = _discriminator_recovers(docs_b, "天枢数据部", "命中率", "0.78", t)
    ck("(b) ★双量纲 78%≠0.78 → fail", (not ok_b) and ans_b == "78%")

    # (c) 一部多负责人歧义:文档里两个负责人 → 判别器答"不确定" → fail
    t = _fake_tracer_factory()
    docs_c = ["项目组本周负责人为王皓,另据交接,该项目负责人为李明,职责待厘清。"]
    ok_c, ans_c = _discriminator_recovers(docs_c, "项目组", "负责人", "王皓", t)
    ck("(c) 多负责人歧义 → 判别器不确定 → fail", (not ok_c) and _is_unsure(ans_c))

    # (d) K=V 句 "X本期「Y」为Z":能还原 → pass(K=V 不自然但忠实,本步只管忠实)
    t = _fake_tracer_factory()
    docs_d = ["2025-01-06 备忘:天枢数据部本期「命中率」为0.78。"]
    ok_d, ans_d = _discriminator_recovers(docs_d, "天枢数据部", "命中率", "0.78", t)
    ck("(d) K=V 句能还原 → pass(忠实)", ok_d and ans_d.startswith("0.78"))

    # ── 死钉③守护:判别器 prompt 里【绝不出现】true_value/gt ──
    all_prompts = "".join(t.seen_prompts) if hasattr(t, "seen_prompts") else ""
    # 复用 (a) 那次的桩痕迹做断言(true_value="0.78" 也在 docs 文本里,无法只查 0.78;
    #   改查 prompt 里没有"应承载/gt/真值/true_value"这类把答案喂进去的字样)
    leaked = any(kw in p for p in (_fake_tracer_factory().seen_prompts or [""]) for kw in ("真值", "gt", "应承载", "true_value"))
    ck("死钉③:判别器 prompt 不含 gt/真值字样", not leaked)
    # 直接验 prompt 构造:user 模板只含 docs+entity+field,不含我们传的 true_value 关键标记
    sample_user = render("discriminate.user", docs="文档正文",
                         queries='[{"key":"q0","entity":"某实体","field":"某字段"}]')
    ck("死钉③:user 模板只含 docs/entity/field", "某实体" in sample_user and "某字段" in sample_user
       and "真值" not in sample_user and "gt" not in sample_user.lower())

    class _BulkTracer:
        def __init__(self):
            self.calls = 0
            self.models = []

        def chat_json(self, _step, _messages, **_kw):
            self.calls += 1
            self.models.append(_kw.get("model"))
            return {"answers": [{"key": "q0", "answer": "甲"},
                                  {"key": "q1", "answer": "乙"}]}

    bulk_tracer = _BulkTracer()
    bulk_answers = _discriminate_many(
        ["一篇同时承载多个字段的文档"],
        [{"key": "q0", "entity": "实体A", "field": "字段A"},
         {"key": "q1", "entity": "实体B", "field": "字段B"}],
        bulk_tracer)
    ck("同组多字段只调用一次盲读者", bulk_tracer.calls == 1
       and bulk_answers == {"q0": "甲", "q1": "乙"}
       and bulk_tracer.models == [config.DISCRIMINATOR_MODEL])

    class _DuplicateKeyTracer:
        def chat_json(self, _step, _messages, **_kw):
            return {"answers": [{"key": "q0", "answer": "甲"},
                                  {"key": "q0", "answer": "乙"}]}

    try:
        _discriminate_many(
            ["正文"],
            [{"key": "q0", "entity": "实体A", "field": "字段A"},
             {"key": "q1", "entity": "实体B", "field": "字段B"}],
            _DuplicateKeyTracer())
        duplicate_rejected = False
    except RuntimeError:
        duplicate_rejected = True
    ck("批量盲读严格拒绝重复或缺失 key", duplicate_rejected)

    class _NoConflictLLM:
        def chat_json(self, *_args, **_kwargs):
            raise AssertionError("L5 传闻不应调用模型")

    class _ConflictWorld:
        conflicts = [{
            "entity": "旧地址解析工件核对",
            "field": "调用状态",
            "session": 2,
            "rumor_value": "执行中",
            "rumor_source": "内部群聊转述",
            "authoritative_source": "官方通报",
        }]

        @staticmethod
        def period_unit():
            return "轮"

    conflict_docs = _render_conflict_docs(_ConflictWorld(), 2, "2025-01-08", _NoConflictLLM())
    ck("L5 传闻由冻结合同确定性渲染且不调模型", len(conflict_docs) == 1)
    ck("L5 传闻逐字承载实体和值并标明低可信",
       all(token in conflict_docs[0]["content"] for token in
           ("旧地址解析工件核对", "调用状态", "执行中", "未经核实", "官方通报")))

    ck("filler 单篇纯正文通过且由代码固定类型",
       _accept_filler_text("外围执行体完成无关归档任务。", {"冻结实体"})
       == [{"type": "背景干扰文档", "content": "外围执行体完成无关归档任务。"}])
    for name, bad, blocked in (
        ("filler 拒绝 JSON 对象壳", {"content": "外围记录"}, set()),
        ("filler 拒绝空正文", "  ", set()),
        ("filler 拒绝冻结专名泄漏", "冻结实体的状态", {"冻结实体"}),
        ("filler 拒绝凭据形态 token", "外围日志记录 sk-exampleToken123456后轮转", set()),
    ):
        try:
            _accept_filler_text(bad, blocked)
            rejected = False
        except RuntimeError:
            rejected = True
        ck(name, rejected)

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"[render self-test] {npass}/{len(checks)} PASS")
    return npass == len(checks)


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(0 if _self_test() else 1)
