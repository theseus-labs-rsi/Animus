"""
eval.multi_system — 多系统记忆评测与有效成绩筛选。

ONE world / 单链:ingest 一次语料,问全部题。对每个记忆系统跑同一套题,
按 line(L1_timeline / L2_relational)与 capability 聚合准确率,观察跨系统排名是否翻转。

系统(在相同公开语料和协议下比较):
  A = SingleShotRAG   : EmbedMemory(top_k=3) 检索 → r1_answer 单轮合成。最朴素 RAG。
  B = FullContextRAG  : 全部 docs(带 [周期|日期] 表头)塞进一个 prompt 一次性作答。
                        仍受上下文预算、注意力与回答模型能力限制。
  C = IterativeRAG    : 先 retrieve → 第一跳 LLM 抽桥实体 → 用桥再 retrieve → 合成。
                        是否比单次检索有收益需要实际测量。

跨系统差异是本批条件下的描述性结果，不能单独证明题目难度、机制必要性或泛化区分度。
执行故障、未决审阅与确定的错误答案分别记录。

跑法:
  ./venv/bin/python -m eval.multi_system                       # 默认 office_v3,A/B/C 全跑
  ./venv/bin/python -m eval.multi_system --smoke               # 烟测(2题×全系统)
  ./venv/bin/python -m eval.multi_system --systems A,B         # 只跑指定系统
  ./venv/bin/python -m eval.multi_system --bench X.json --corpus Y.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
import threading
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from eval.memory_interface import EmbedMemory, _chunk, build_embed_memory  # noqa: F401  re-export: 定义已移到 memory_interface,避免内置系统 import legacy driver
from eval.embed_cache import cached_embed, cache_size
from eval import qa_cache
from eval.question_filter import export_filtered_benchmark, validate_options as validate_filter_options
from eval.baseline_r1 import unified_answer
from eval.judge import judge, judge_record, judgement, is_judgeable, gold_display, judge_spec, classify_refusal, literal_match
from eval.grading import JUDGE_VERSION, is_scored, requires_semantic_grading
from eval.reassessment import ReassessmentTracker, closure_path, VERSION as REASSESSMENT_VERSION
from eval.provenance import (load_visible_corpus, load_public_protocol,
                             make_evaluation_context, record_provenance, valid_context, protocol_scoring_policy)
from eval.public_context import header, build_full_context
from llm_trace import trace_scope
from contextlib import nullcontext

# ── 默认评测集(office_v3) ───────────────────────────────────────────────────
DEFAULT_BENCH = ROOT / "output" / "factory_v2_office_v3" / "06_questions.json"
DEFAULT_CORPUS = ROOT / "output" / "factory_v2_office_v3" / "05_corpus.json"
OUT_JSON = ROOT / "output" / "eval" / "multi_system_office_v3.json"
OUT_REPORT = ROOT / "output" / "eval" / "multi_system_report.md"

# 全量语料约 10万字符(~120k token),模型实测能吃下;留个安全闸,超大语料才截断。
FULLCTX_CHAR_BUDGET = 120_000
TOP_K = 3
QA_MAX_TOKENS = 2048
WORKERS = 6

# ── 进度探针(纯旁路,写 _progress.json 供 eval_monitor 只读轮询)──────────────
class _EvalProbe:
    FEED_CAP = 40

    def __init__(self, output_dir=None):
        self._lk = threading.RLock()
        self._d = {}
        self._feed = []
        self.eval_id = f"eval_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.run_dir = Path(output_dir).resolve() if output_dir is not None else ROOT / "output" / "eval" / self.eval_id
        self._progress = self.run_dir / "_progress.json"
        self.write_error = None

    def _flush(self):
        with self._lk:
            tmp = None
            try:
                self._d["ts"] = time.time()
                self._d["eval_id"] = self.eval_id
                self._d["feed"] = self._feed[-self.FEED_CAP:]
                self.run_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                        dir=self.run_dir, prefix="_progress-", suffix=".tmp", delete=False) as handle:
                    tmp = Path(handle.name)
                    json.dump(self._d, handle, ensure_ascii=False, indent=1)
                # rename cannot overwrite an existing destination on Windows.
                for attempt in range(5):
                    try:
                        os.replace(tmp, self._progress)
                        break
                    except PermissionError:
                        # Windows readers may briefly hold the destination open.
                        if attempt == 4:
                            raise
                        time.sleep(0.01 * (2 ** attempt))
                self.write_error = None
            except Exception as exc:
                error = type(exc).__name__
                if self.write_error != error:
                    print(f"[eval progress] 写入失败 ({error})：{self._progress}", file=sys.stderr)
                self.write_error = error
            finally:
                if tmp is not None:
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass

    def start(self, **kw):
        with self._lk:
            self._d = {"status": "running", "started_ts": time.time(), **kw,
                    "current": "", "embed": {},
                    "sys": {s: {"status": "pending", "done": 0, "judged": 0,
                                "correct": 0, "j_real": 0,
                                "total": kw.get("n_selected", kw.get("n_judgeable", 0))}
                            for s in kw.get("systems", [])}}
            self._feed = []
            self._flush()

    def embed_start(self, total):
        with self._lk:
            self._d["embed"] = {"status": "running", "ts0": time.time(), "done": 0, "total": total}
            self._flush()

    def embed_tick(self, done):
        with self._lk:
            self._d["embed"]["done"] = max(done, self._d["embed"].get("done", 0))
            self._flush()

    def embed_done(self, n_chunks, elapsed):
        with self._lk:
            self._d["embed"] = {"status": "done", "n_chunks": n_chunks, "elapsed_s": round(elapsed, 1)}
            self._flush()

    def sys_start(self, name, total):
        with self._lk:
            self._d["current"] = name
            self._d["sys"][name] = {"status": "answering", "done": 0, "judged": 0,
                                 "correct": 0, "j_real": 0, "total": total, "ts0": time.time()}
            self._flush()

    def answer_tick(self, name, i):
        with self._lk:
            self._d["sys"][name]["done"] = max(i, self._d["sys"][name]["done"])
            self._flush()

    def judge_start(self, name):
        with self._lk:
            self._d["sys"][name]["status"] = "judging"
            self._flush()

    def judge_tick(self, name, i, rec):
        with self._lk:
            sd = self._d["sys"][name]
            sd["judged"] = max(i, sd["judged"])
            if is_scored(rec):
                sd["j_real"] = sd.get("j_real", 0) + 1
                if rec["correct"]:
                    sd["correct"] = sd.get("correct", 0) + 1
            else:
                sd["incomplete"] = sd.get("incomplete", 0) + 1
            self._feed.append({
                "s": name, "i": i, "ln": (rec.get("line") or "")[:2],
                "cap": rec.get("capability", ""),
                "ok": rec.get("correct"),
                "verdict": (rec.get("judgement") or {}).get("verdict"),
                "pred": str(rec.get("pred", ""))[:40],
                "gold": str(rec.get("gold_set", ""))[:40],
            })
            self._flush()

    def sys_done(self, name, agg, elapsed):
        with self._lk:
            sd = self._d["sys"][name]
            sd["status"] = "done"
            sd["elapsed_s"] = round(elapsed, 1)
            sd["acc"] = agg["overall"].get("acc")
            sd["scored"] = agg["overall"]["n"]
            sd["j_real"] = agg["overall"]["n"]
            sd["correct"] = agg["overall"]["correct"]
            sd["unscored"] = agg.get("n_unscored", agg.get("n_incomplete", 0))
            sd["incomplete"] = sd["unscored"]  # legacy display alias
            sd["by_line"] = {ln: d["acc"] for ln, d in agg.get("by_line", {}).items()
                         if d.get("acc") is not None}
            self._flush()

    def done(self, disc=None):
        with self._lk:
            self._d["status"] = "done"
            self._d["current"] = ""
            self._d["elapsed_s"] = round(time.time() - self._d.get("started_ts", time.time()), 1)
            if disc:
                self._d["disc"] = disc
            self._flush()


# 判分单一真源在 eval/judge.py:judge_spec(按声明的 capability 分派 value/refusal/order)。
# (旧 gold_strings / SKIP_CAPABILITIES / L1_CAPS_ORDER 已废除,避免平行旧真源。)


# ─────────────────────────────────────────────────────────────────────────────
# 语料加载 + 公平 ingest(照搬 run_eval:每片段自带 [周期|日期] 表头)
# ─────────────────────────────────────────────────────────────────────────────
def load_corpus(corpus_path: Path) -> list:
    """返回按 session_id 升序的 [(sid, date, content), ...](doc 粒度)。"""
    return load_visible_corpus(corpus_path)


def load_protocol(about_path: Path) -> str:
    """读 00_about.json 的 answer_protocol,渲染成给被测系统的【答题约定】文本。
    这是随题库交付的作答契约；v5 区分未记录、已停统与范围外，旧协议保持兼容。
    三系统同等注入 → 公平;给规则不等于给答案。读不到则返回空串(不注入)。"""
    return load_public_protocol(about_path)


# ─────────────────────────────────────────────────────────────────────────────
# 单系统跑评(并行)
# ─────────────────────────────────────────────────────────────────────────────
def execution_failure(stage: str, exc: Exception) -> dict:
    """Preserve typed operation evidence without serializing provider secrets."""
    from eval.memory_systems.base import MemoryExecutionError
    if isinstance(exc, MemoryExecutionError):
        status = "incomplete" if exc.code in {"completion_unverified", "dependency_completion_unverified"} else "error"
        return {"status": status, **exc.as_dict()}
    return {"status": "error", "stage": stage, "code": "operation_failed",
            "details": {"cause_type": type(exc).__name__}}


def prepare_systems(names, sessions, factory, on_progress=None):
    """A failed ingest is a failed system run; other systems still get records."""
    systems, executions = {}, {}
    shared_embed, shared_execution = None, None
    for name in names:
        receipts, stage = [], "initialize"
        try:
            if name in ("A", "C") and shared_execution is not None:
                if shared_execution["status"] != "ok":
                    systems[name], executions[name] = None, dict(shared_execution)
                    continue
                systems[name] = factory(name, embed_mem=shared_embed)
                executions[name] = {**shared_execution, "shared_embedding": True}
                continue
            instance = factory("A" if name == "C" else name)
            stage = "ingest"
            for session in sessions:
                receipt = instance.ingest_session(session)
                if not isinstance(receipt, dict) or receipt.get("status") != "ok":
                    from eval.memory_systems.base import MemoryExecutionError
                    raise MemoryExecutionError("ingest", "invalid_receipt", {"session_id": session["session_id"]})
                if type(receipt.get("n_docs")) is not int or receipt.get("n_docs") != len(session["docs"]):
                    from eval.memory_systems.base import MemoryExecutionError
                    raise MemoryExecutionError("ingest", "partial_write", {
                        "session_id": session["session_id"], "completed_docs": receipt.get("n_docs"),
                        "requested_docs": len(session["docs"])})
                receipts.append({"session_id": session["session_id"], **receipt})
            stage = "finalize_ingest"
            finalization = instance.finalize_ingest(on_progress=on_progress)
            execution = {"status": "ok", "stage": "ready", "receipts": receipts,
                         "requested_docs": sum(len(s["docs"]) for s in sessions)}
            if isinstance(finalization, dict):
                execution["finalization"] = finalization
            if name in ("A", "C"):
                shared_embed, shared_execution = instance._mem, execution
                if name == "C":
                    stage = "initialize"
                    instance = factory(name, embed_mem=shared_embed)
            systems[name], executions[name] = instance, execution
        except Exception as exc:
            execution = {**execution_failure(stage, exc), "receipts": receipts,
                         "requested_docs": sum(len(s["docs"]) for s in sessions)}
            systems[name], executions[name] = None, execution
            if name in ("A", "C") and shared_execution is None:
                shared_execution = execution
    return systems, executions


def run_system(name: str, questions: list, sys_instance, workers: int = WORKERS,
               verbose: bool = True, protocol: str = "",
               probe=None, bench_id: str = None, resume: bool = True,
               cache_context: dict | None = None, execution: dict | None = None,
               judge_fn=None, grading_context: dict | None = None, trace_path=None,
               reassessment: ReassessmentTracker | None = None,
               answer_fn=None, solver_identity: dict | None = None) -> list:
    """对一个系统跑全部题,返回 records(每题一条,含 pred/correct/judgeable)。
    sys_instance:已 ingest 好的 MemorySystem 实例。
    protocol:随题库交付的【答题约定】,同等注入三系统的作答提示。
    bench_id+resume:QA 断点续传——已判过的题从盘加载、跳过(干净结果才入盘,错的下轮重试)。"""
    supplied = cache_context or {}
    published_policy = protocol_scoring_policy(protocol)
    if any(requires_semantic_grading(q) for q in questions) and published_policy is None:
        raise ValueError("Process questions require their published A policy and semantic grading")
    if published_policy is not None:
        from eval.semantic_judge import resolve_scoring_config
        scoring = getattr(judge_fn, "scoring_configuration", None)
        if (not isinstance(scoring, dict) or scoring.get("scoring_policy") != published_policy
                or scoring != resolve_scoring_config(published_policy, scoring.get("grading_method"))):
            raise ValueError("The public A policy requires its matching semantic judge; legacy grading is not allowed")
    explicit_solver = None
    if answer_fn is not None:
        if not callable(answer_fn) or not isinstance(solver_identity, dict):
            raise ValueError("An injected answer function requires an explicit JSON solver identity")
        try:
            explicit_solver = json.loads(json.dumps(solver_identity, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("Solver identity must be finite JSON") from exc
        for key in ("model", "transport_fingerprint", "answer_prompt_hash", "implementation_version"):
            if not isinstance(explicit_solver.get(key), str) or not explicit_solver[key].strip():
                raise ValueError("Missing explicit solver identity: " + key)
        if type(explicit_solver.get("max_tokens")) is not int or explicit_solver["max_tokens"] < 1:
            raise ValueError("An injected solver requires a positive token budget")
    elif solver_identity is not None:
        raise ValueError("solver_identity requires answer_fn")
    evaluation_context = supplied.get("evaluation_context")
    if valid_context(evaluation_context) and evaluation_context["protocol_hash"] != qa_cache.digest(protocol):
        raise ValueError("Evaluation identity does not match the actual public protocol")
    adapter_configuration = {}
    if callable(getattr(type(sys_instance), "evaluation_config", None)):
        adapter_configuration = sys_instance.evaluation_config()
        if not isinstance(adapter_configuration, dict):
            raise ValueError("Adapter evaluation_config must return a JSON object")
    solver_context = {**supplied.get("solver", {}), "model": getattr(config, "MODEL", None), "protocol": protocol,
               "system_class": type(sys_instance).__module__ + "." + type(sys_instance).__qualname__,
               "adapter_configuration_hash": qa_cache.digest(adapter_configuration),
               "top_k": TOP_K, "qa_max_tokens": QA_MAX_TOKENS, "fullctx_budget": FULLCTX_CHAR_BUDGET,
               "api_endpoint_hash": qa_cache.digest(getattr(config, "BASE_URL", None)),
               "system_env_hash": qa_cache.digest({k: v for k, v in os.environ.items()
                   if k.startswith(("MEM0_", "MEMOS_", "ZEP_", "SIMPLEMEM_", "AMEM_", "EMBED_", "LLM_", "INGEST_LLM_"))})}
    # Preserve old callers' corpus/source keys but keep judge identity separate.
    solver_context.update({k: v for k, v in supplied.items() if k not in {"solver", "grading", "evaluation_context"}})
    if explicit_solver is not None:
        # The callable owns its model/prompt/profile. Keep the actual adapter
        # and protocol authoritative, and never silently substitute config.MODEL.
        solver_context.update(model=explicit_solver["model"], protocol=protocol,
            qa_max_tokens=explicit_solver["max_tokens"], answer_identity=explicit_solver,
            system_class=type(sys_instance).__module__ + "." + type(sys_instance).__qualname__,
            adapter_configuration_hash=qa_cache.digest(adapter_configuration))
    callable_grading = getattr(judge_fn, "cache_context", {}) if judge_fn else {}
    grade_cache_enabled = judge_fn is None or bool(callable_grading or grading_context)
    context = {"solver": solver_context, "evaluation_context": evaluation_context,
               "grading": {"version": JUDGE_VERSION, "model": getattr(config, "JUDGE_MODEL", getattr(config, "MODEL", None)),
                           "judge_source_hash": qa_cache.digest((ROOT / "eval/judge.py").read_text(encoding="utf-8")),
                           **supplied.get("grading", {}), **callable_grading, **(grading_context or {})}}
    if reassessment is not None:
        if judge_fn is None:
            raise ValueError("Reassessment closure only applies to semantic grading")
        reassessment.validate_context(evaluation_context, context["grading"].get("reference_review_hash"))
        # Old semantic caches predate durable cross-system dispute closure. Keep
        # their predictions reusable, but require fresh grading under this policy.
        context["grading"]["reassessment_policy"] = REASSESSMENT_VERSION
    # A caller that cannot identify its corpus must not reuse cached answers.
    identified = valid_context(evaluation_context) or solver_context.get("corpus_hash")
    if (adapter_configuration.get("configuration_status") == "undeclared"
            and not supplied.get("solver", {}).get("adapter_config")):
        identified = False
    bid = qa_cache.bench_id(questions, context) if bench_id and identified else None
    done = qa_cache.load(bid, name) if (resume and bid and grade_cache_enabled) else {}
    predictions = qa_cache.load_predictions(bid, name) if (resume and bid) else {}
    if verbose:
        print(f"\n[{name}] 提问 {len(questions)} 题 ..."
              + (f"(断点续传:已有 {len(done)} 题,本轮跳过)" if done else ""))

    # ── 每题的求解函数(题与题独立 → pmap 并行) ──
    def solve(q: dict) -> dict:
        prediction_key = qa_cache.qhash(q, context, grading=False)
        mode = "semantic" if judge_fn else judge_spec(q)[0]
        rec = {**q,
            "qid": q.get("qid"),
            "line": q.get("line"), "capability": q.get("capability"),
            "question": q["question"], "gt": q.get("gt"),
            "aux": q.get("aux"),                             # ★L6 透传:判分需 aux.lure.value(吐诱饵=判错)
            "strict_scoring": q.get("strict_scoring"),       # 明星题全原子合同必须穿透真实评测调用链
            "gold_set": q.get("reference_proposal", q.get("gold", q.get("gt"))) if judge_fn else gold_display(q), "mode": mode,
            "judgeable": bool(q.get("question")) if judge_fn else is_judgeable(q),
            "execution_status": "ok",
            "evaluation_scope": "research_only" if judge_fn else "legacy",
        }
        # A missing reference and an explicit null proposal are different inputs.
        # Do not manufacture legacy fields on an open semantic task.
        if judge_fn:
            for key in ("gt", "aux", "strict_scoring"):
                if key not in q:
                    rec.pop(key, None)
        if valid_context(evaluation_context):
            rec["evaluation_provenance"] = record_provenance(q, evaluation_context)
        if explicit_solver is not None:
            rec["solver_identity"] = json.loads(json.dumps(explicit_solver))
        if execution and execution.get("status") != "ok":
            rec.update(pred="", execution_status=execution.get("status", "error"), execution=execution,
                       error="memory preparation not completed")
            return rec
        if prediction_key in predictions:
            rec.update({k: v for k, v in predictions[prediction_key].items()
                        if k in ("pred", "bridge_extracted", "retrieved_context_hash")})
            rec["_prediction_resumed"] = True
            return rec
        stage = "retrieve"
        try:
            with trace_scope(trace_path, "eval.solve", metadata={"system": name, "question_hash": prediction_key}) if trace_path else nullcontext():
                retrieved_context = sys_instance.retrieve(q["question"], top_k=TOP_K)
                if not isinstance(retrieved_context, str):
                    raise TypeError("retrieve must return text on successful execution")
                stage = "answer"
                if explicit_solver is None:
                    rec["pred"] = unified_answer(q["question"], retrieved_context, protocol=protocol,
                                                 max_tokens=QA_MAX_TOKENS)
                else:
                    rec["retrieved_context_hash"] = qa_cache.digest(retrieved_context)
                    # Only public strings cross this boundary, never the
                    # reference proposal, seed or private reviewer messages.
                    prediction = answer_fn(q["question"], retrieved_context, protocol=protocol,
                                           max_tokens=explicit_solver["max_tokens"])
                    if not isinstance(prediction, str) or not prediction.strip():
                        raise ValueError("Injected answer function must return nonempty text")
                    rec["pred"] = prediction
            diag = sys_instance.get_diagnostics()
            if diag.get("bridge"):
                rec["bridge_extracted"] = diag["bridge"]
            if bid:
                qa_cache.append_prediction(bid, name, {**rec, "_qh": prediction_key})
        except Exception as e:
            rec["pred"] = ""
            rec["execution"] = execution_failure(stage, e)
            rec["execution_status"] = rec["execution"]["status"]
            rec["error"] = f"{stage}:{rec['execution']['code']}"
        return rec

    if probe:
        probe.sys_start(name, len(questions))
        _solve0, _slk, _sn = solve, threading.Lock(), [0]
        def solve(q, _f=_solve0):
            r = _f(q)
            with _slk:
                _sn[0] += 1
                probe.answer_tick(name, _sn[0])
            return r

    records = config.pmap(solve, questions, workers=workers)

    # ── 3) 判分(也并行;失败标记但不崩) ──
    def do_judge(rec: dict) -> dict:
        pred = rec["pred"]
        qh = qa_cache.qhash(rec, context, prediction=pred)
        rec["_qh"] = qh
        if reassessment is not None and reassessment.is_disputed(rec):
            if qh in done and rec.get("execution_status") == "ok":
                rec.update(judgement=done[qh]["judgement"], correct=done[qh]["correct"], _resumed=True)
            reassessment.quarantine_record(rec)
            return rec
        if rec.get("execution_status") != "ok":
            grade = judgement("unjudgeable" if rec.get("execution_status") == "incomplete" else "error",
                              "execution", rec.get("error") or "solver execution not completed")
        elif qh in done:
            cached = done[qh]
            # Keep current public identity/labels while reusing only this answer's grade.
            rec.update(judgement=cached["judgement"], correct=cached["correct"], _resumed=True)
            if reassessment is not None:
                reassessment.quarantine_record(rec)
            return rec
        elif not rec["judgeable"]:
            grade = judgement("unjudgeable", "contract", "missing or unsupported grading contract")
        else:
            try:
                with trace_scope(trace_path, "eval.judge", metadata={"system": name, "question_hash": qh}) if trace_path else nullcontext():
                    grade = judge_fn(rec, pred) if judge_fn else judge_record(rec, pred, use_llm=True)
                if not judge_fn and rec["capability"] == "L6_refusal":
                    lure = ((rec.get("aux") or {}).get("lure") or {}).get("value")
                    rec["refusal_bucket"] = classify_refusal(pred, lure)
                if not judge_fn and rec["capability"] == "L2_multihop":
                    # Reuse the primary grade; never issue a second judge call.
                    rec["partial"] = (1.0 if grade["correct"] is True else
                                      0.5 if grade["correct"] is False and literal_match(pred, [(rec.get("aux") or {}).get("bridge")]) else
                                      0.0 if grade["correct"] is False else None)
            except Exception as e:
                grade = judgement("error", "judge_exception", f"{type(e).__name__}:{str(e)[:160]}")
        rec["judgement"] = grade
        rec["correct"] = grade["correct"]
        if grade["verdict"] == "error":
            rec["judge_error"] = grade["reason"]
        if reassessment is not None:
            reassessment.observe([rec], system=name)
            reassessment.quarantine_record(rec)
        # 干净结果才入盘续传;带 error/judge_error 的(端点抖动所致)不存 → 下轮重试
        if bid and grade_cache_enabled and is_scored(rec):
            qa_cache.append(bid, name, rec)
        return rec

    if probe:
        probe.judge_start(name)
        _jdg0, _jlk, _jn = do_judge, threading.Lock(), [0]
        def do_judge(rec, _f=_jdg0):
            r = _f(rec)
            with _jlk:
                _jn[0] += 1
                probe.judge_tick(name, _jn[0], r)
            return r

    records = config.pmap(do_judge, records, workers=workers)

    if hasattr(sys_instance, 'trunc_info'):
        ti = sys_instance.trunc_info
        for r in records:
            r["_fullctx"] = ti

    if verbose:
        for r in records:
            mark = "∅" if r["correct"] is None else ("✓" if r["correct"] else "✗")
            extra = ""
            if "bridge_extracted" in r:
                extra = f"  [桥={r['bridge_extracted'][:12]!r}]"
            gold = r["gold_set"] if r["judgeable"] else "(不可判分)"
            print(f"  {mark} [{str(r.get('line') or 'semantic')[:2]}/{r.get('capability') or 'open_task'}] "
                  f"pred={str(r['pred'])[:28]!r} gold={gold}{extra}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 聚合 + 报表
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(records: list) -> dict:
    """准确率只统计已有有效成绩的记录；未计分原因另列。"""
    jr = [r for r in records if r.get("judgeable") and is_scored(r)]
    n_unjudge = len(records) - len(jr)

    def acc(rs):
        rs = [r for r in rs if r.get("judgeable") and is_scored(r)]
        if not rs:
            return {"n": 0, "correct": 0, "acc": None}
        c = sum(1 for r in rs if r["correct"])
        return {"n": len(rs), "correct": c, "acc": round(c / len(rs), 3)}

    by_line = {ln: acc([r for r in jr if (r.get("line") or "semantic") == ln])
               for ln in sorted({r.get("line") or "semantic" for r in jr})}
    by_cap = {cap: acc([r for r in jr if (r.get("capability") or "open_task") == cap])
              for cap in sorted({r.get("capability") or "open_task" for r in jr})}
    return {
        "overall": acc(jr),
        "by_line": by_line,
        "by_capability": by_cap,
        "n_scored": len(jr),
        "n_unscored": n_unjudge,
        "n_unjudgeable": sum(1 for r in records if (r.get("judgement") or {}).get("verdict") == "unjudgeable"),
        "n_incomplete": n_unjudge,
        "count_semantics": {"n_incomplete": "legacy_alias_for_n_unscored",
                            "n_unjudgeable": "judgement_verdict_unjudgeable_only"},
        "n_errors": sum(1 for r in records if (r.get("judgement") or {}).get("verdict") == "error"),
        "n_execution_errors": sum(1 for r in records if r.get("execution_status") == "error"),
        "n_execution_incomplete": sum(1 for r in records if r.get("execution_status") == "incomplete"),
        "n_uncertain": sum(1 for r in records if (r.get("judgement") or {}).get("verdict") == "uncertain"),
        "execution_completion_rate": sum(r.get("execution_status", "ok") == "ok" for r in records) / len(records) if records else None,
    }


def close_semantic_reassessment(results, tracker):
    """Close cross-system disputes before final accuracy, export or easy filtering."""
    closure = tracker.close({system: result["records"] for system, result in results.items()})
    for result in results.values():
        result["agg"] = aggregate(result["records"])
    return closure


def selection_metadata(n_total: int, n_eligible: int, n_selected: int, results: dict) -> dict:
    """Separate input selection counts from each system's actual scored records."""
    return {"n_total": n_total, "n_eligible": n_eligible, "n_selected": n_selected,
            "n_excluded": n_total - n_eligible, "n_not_selected": n_eligible - n_selected,
            "n_scored_by_system": {s: r["agg"]["overall"]["n"] for s, r in results.items()},
            "n_unscored_by_system": {s: r["agg"].get("n_unscored", r["agg"].get("n_incomplete", 0))
                                     for s, r in results.items()},
            "n_judgeable": n_eligible, "n_unjudgeable": n_total - n_eligible,
            "count_semantics": {"n_judgeable": "legacy_alias_for_n_eligible_not_scored",
                                "n_unjudgeable": "legacy_alias_for_n_excluded_before_run",
                                "n_not_selected": "eligible_items_not_selected_for_this_run"}}


# 终端 ANSI 颜色(per-capability 着色:绿=高,黄=中,红=低)
def _color_acc(a):
    if a is None:
        return "  -  "
    s = f"{a:5.0%}"
    if a >= 0.75:
        return f"\033[92m{s}\033[0m"   # 绿
    if a >= 0.40:
        return f"\033[93m{s}\033[0m"   # 黄
    return f"\033[91m{s}\033[0m"       # 红


SYSTEM_LABELS = {
    "A": "SingleShotRAG", "B": "FullContextRAG", "C": "IterativeRAG",
    "mem0": "Mem0", "zep": "Zep", "memos": "MemOS", "amem": "A-Mem",
}
SYSTEM_DESC = {
    "A": "EmbedMemory(top_k=3) 检索 → r1 单轮合成。最朴素 RAG;紧检索易漏被隐藏的多跳桥文档/晚期证据。",
    "B": "全部 docs(带 [周期|日期] 表头)拼入上下文后作答；受上下文预算与模型能力限制。",
    "C": "retrieve → LLM 抽桥实体并改写子问 → 用桥再 retrieve → 合并两跳证据作答。",
    "mem0": "Mem0(Qdrant + LLM fact extraction)。事实三元组自动提取 + 向量检索。",
    "zep": "Zep 时序知识图。对话→时序 fact graph → 语义搜索。支持 CE 自托管或 Cloud。",
    "memos": "MemOS(MemTensor)。多层记忆架构(事实 + 偏好 + 对话)。支持自托管或 Cloud。",
    "amem": "A-Mem(Zettelkasten 图记忆)。LLM 分析→结构化 note(keywords/tags/links)→向量+图扩展检索。",
}
# 七线展示顺序(report/表格)。短码 = ln.split('_')[0](L1..L7)。
LINE_ORDER = ["L1_timeline", "L2_relational", "L3_process", "L4_preference",
              "L5_conflict", "L6_refusal", "L7_consolidation"]


def _present_lines(results: dict, sys_names: list) -> list:
    """出现在结果里的 line,按 LINE_ORDER 排;未登记的兜底追加。"""
    seen = set()
    for s in sys_names:
        seen |= set(results[s]["agg"]["by_line"].keys())
    ordered = [ln for ln in LINE_ORDER if ln in seen]
    return ordered + [ln for ln in sorted(seen) if ln not in LINE_ORDER]


def discrimination_summary(results: dict, sys_names: list) -> dict:
    """至少两个完整系统的描述性分数差；不推断泛化区分度或题库难度。

    调用方须保证同一冻结题集、公开语料和协议（main 使用同批输入）。
    这个展示函数不认证外部传入结果的来源或跨批次可比性。
    """
    sys_names = list(dict.fromkeys(sys_names))
    def la(s, ln):
        d = results[s]["agg"]["by_line"].get(ln)
        return d["acc"] if d and d["acc"] is not None else None

    lines = _present_lines(results, sys_names)
    overall = {s: results[s]["agg"]["overall"]["acc"] for s in sys_names}
    incomplete = any(results[s]["agg"].get("n_unscored", results[s]["agg"].get("n_incomplete", 0))
                     or overall[s] is None for s in sys_names)
    if incomplete or len(sys_names) < 2:
        return {"status": "incomplete" if incomplete else "insufficient_systems",
                "n_systems": len(sys_names),
                "reason": "unscored_records" if incomplete else "requires_at_least_two_distinct_systems",
                "spreads": {ln: None for ln in lines}, "ranking": [],
                "overall": overall, "ov_spread": None, "max_line_spread": None,
                "headroom": None, "best": None, "discriminates": None, "lines": lines}
    spreads = {}
    for ln in lines:
        accs = [la(s, ln) for s in sys_names]
        accs = [a for a in accs if a is not None]
        spreads[ln] = (max(accs) - min(accs)) if len(accs) >= 2 else None

    ranking = sorted(sys_names, key=lambda s: overall[s], reverse=True)
    ov_vals = [overall[s] for s in sys_names]
    ov_spread = (max(ov_vals) - min(ov_vals)) if len(ov_vals) >= 2 else 0.0
    best = max(ov_vals) if ov_vals else 0.0
    headroom = 1.0 - best

    max_line_spread = max((v for v in spreads.values() if v is not None), default=0.0)
    discriminates = (ov_spread >= 0.10) or (max_line_spread >= 0.20)

    return {"status": "complete", "n_systems": len(sys_names),
            "spreads": spreads, "ranking": ranking, "overall": overall,
            "ov_spread": ov_spread, "max_line_spread": max_line_spread,
            "headroom": headroom, "best": best, "discriminates": discriminates,
            "lines": lines}


def print_table(results: dict, sys_names: list):
    lines = _present_lines(results, sys_names)
    short = [ln.split("_")[0] for ln in lines]
    print("\n" + "=" * (20 + 8 * len(lines) + 8))
    print("=== 多系统记忆评测(七线 | bench={} 系统={}) ===".format(
        "/".join(short), ",".join(sys_names)))
    print("=" * (20 + 8 * len(lines) + 8))

    def cell(d):
        return _color_acc(d.get("acc") if d else None)

    hdr = f"{'系统':<16}" + "".join(f"{c:<8}" for c in short) + f"{'overall':<10}"
    print(hdr)
    print("-" * (16 + 8 * len(lines) + 10))
    for s in sys_names:
        agg = results[s]["agg"]
        row = f"{s+'='+SYSTEM_LABELS.get(s, s):<16}"
        for ln in lines:
            row += f"{cell(agg['by_line'].get(ln)):<17}"   # +9 for ANSI codes
        row += f"{cell(agg['overall']):<17}"
        print(row)

    print("\n计分覆盖（各系统独立统计）:")
    for s in sys_names:
        agg = results[s]["agg"]
        print(f"  {s}: 已计分={agg['overall']['n']}，未计分="
              f"{agg.get('n_unscored', agg.get('n_incomplete', 0))}")

    # per-capability(全 capability,着色)
    caps = sorted({c for s in sys_names for c in results[s]["agg"]["by_capability"]})
    print("\n--- per-capability accuracy(着色:绿≥75% 黄≥40% 红<40%)---")
    print(f"{'系统':<12}" + "".join(f"{c[:13]:<14}" for c in caps))
    for s in sys_names:
        bc = results[s]["agg"]["by_capability"]
        row = f"{s:<12}"
        for c in caps:
            row += f"{_color_acc((bc.get(c) or {}).get('acc')):<23}"
        print(row)

    # 区分度
    ds = discrimination_summary(results, sys_names)
    print("\n--- 跨系统分数比较 ---")
    if ds["status"] == "incomplete":
        print("  待复核：存在未计分记录，暂不比较系统排名或判断题库难度。")
        return
    if ds["status"] == "insufficient_systems":
        print("  未评估跨系统区分：至少需要两个不同系统；本次只展示各系统成绩。")
        return
    print("  系统分数（降序）: " + " / ".join(
        f"{s}({ds['overall'][s]:.0%})" for s in ds["ranking"]))
    print(f"  总分离差(max-min)= {ds['ov_spread']:.0%}  |  最高分余量(1-best)= {ds['headroom']:.0%}")
    print("  逐线离差: " + "  ".join(
        f"{ln.split('_')[0]}=" + (f"{v:.0%}" if v is not None else "无可比数据")
        for ln, v in ds["spreads"].items()))
    print("  描述: " + ("本批分数差达到描述性阈值（总分≥10%或逐线≥20%）。"
                        if ds["discriminates"] else
                        "本批分数差未达到描述性阈值。"))
    print("  这些差异不单独证明题库难度或泛化区分度。")


def write_report(results: dict, sys_names: list, meta: dict, out_path=None):
    """写 markdown 报告:七线主表 + per-capability + 区分度 + 解读 + caveat。"""
    out_path = out_path or OUT_REPORT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def mdacc(d):
        a = d.get("acc") if d else None
        return f"{a:.0%} (n={d['n']})" if (d and a is not None) else "— (n=0)"

    lines = _present_lines(results, sys_names)
    ds = discrimination_summary(results, sys_names)

    L = []
    L.append("# 多系统记忆评测报告 — 七线\n")
    if (meta.get("release") or {}).get("override"):
        L.append("> 历史研究模式：显式绕过发布资格，以下仅为实验观察，不是正式 benchmark 成绩。\n")
    L.append(f"- 判分版本：{meta.get('judge_version', JUDGE_VERSION)}；裁判模型：{meta.get('judge_model', '未记录')}；故障、未决及不可判分项不进入能力分母。")
    L.append(f"- 评测集:`{meta['bench']}`：输入 {meta['n_total']} 题；"
             f"本次选取 {meta.get('n_selected', '未记录')} 题；"
             f"输入预排除 {meta.get('n_excluded', '未记录')} 题。实际计分数量按系统单列。")
    L.append(f"- 语料:`{meta['corpus']}`({meta['n_sessions']} sessions / "
             f"{meta['n_docs']} docs / {meta['corpus_chars']} 字符)")
    L.append(f"- 协议注入:{'是(随题库交付的答题约定已同等注入参评系统)' if meta.get('protocol_injected') else '否'}")
    L.append(f"- 模型:`{config.MODEL}`(DMXAPI)| QA/judge temperature=0\n")

    L.append("## 系统")
    for s in sys_names:
        L.append(f"- **{s} = {SYSTEM_LABELS.get(s, s)}** — {SYSTEM_DESC.get(s, '')}")
    L.append("")

    L.append("## 计分覆盖\n")
    L.append("| 系统 | 已计分 | 未计分 | 不可判分裁决 | 未决裁决 | 错误状态 |")
    L.append("|---|---|---|---|---|---|")
    for s in sys_names:
        agg = results[s]["agg"]
        L.append(f"| {s} | {agg['overall']['n']} | "
                 f"{agg.get('n_unscored', agg.get('n_incomplete', 0))} | "
                 f"{agg.get('n_unjudgeable', 0)} | {agg.get('n_uncertain', 0)} | {agg.get('n_errors', 0)} |")
    L.append("\n未计分包含执行故障、完成状态不确定及未形成有效裁决的记录；它不等于输入预排除数，也不计作答错。错误状态包括执行错误和判分错误。\n")

    # 主表:line × 系统
    short = [ln.split("_")[0] for ln in lines]
    L.append("## 主表:line × 系统 accuracy\n")
    L.append("| 系统 | " + " | ".join(short) + " | overall |")
    L.append("|" + "---|" * (len(lines) + 2))
    for s in sys_names:
        agg = results[s]["agg"]
        cells = [mdacc(agg["by_line"].get(ln)) for ln in lines]
        L.append(f"| {s}={SYSTEM_LABELS.get(s, s)} | " + " | ".join(cells)
                 + f" | {mdacc(agg['overall'])} |")
    L.append("")

    # per-capability
    caps = sorted({c for s in sys_names for c in results[s]["agg"]["by_capability"]})
    L.append("## per-capability accuracy\n")
    L.append("| 系统 | " + " | ".join(caps) + " |")
    L.append("|" + "---|" * (len(caps) + 1))
    for s in sys_names:
        bc = results[s]["agg"]["by_capability"]
        row = [s] + [mdacc(bc.get(c)) for c in caps]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # 区分度
    L.append("## 跨系统分数比较\n")
    if ds["status"] == "complete":
        L.append("- 系统分数（降序）: " + " / ".join(f"{s}({ds['overall'][s]:.0%})" for s in ds["ranking"]))
        L.append(f"- 总分离差(max−min)= **{ds['ov_spread']:.0%}**;最高分余量(1−best)= **{ds['headroom']:.0%}**"
                 f"(best={ds['best']:.0%})")
        L.append("- 逐线离差: " + "; ".join(
            f"{ln.split('_')[0]}=" + (f"{v:.0%}" if v is not None else "无可比数据") for ln, v in ds["spreads"].items()))
        L.append("- **描述**: " + ("本批分数差达到描述性阈值（总分≥10%或逐线≥20%）。"
                                  if ds["discriminates"] else "本批分数差未达到描述性阈值。"))
    elif ds["status"] == "insufficient_systems":
        L.append("- **未评估跨系统区分**：至少需要两个不同系统；本次只展示各系统成绩。")
    else:
        L.append("- **待复核**：存在未计分记录，暂不比较系统排名或判断题库难度。")
    L.append("")

    # 解读 + caveat
    L.append("## 解读\n")
    for s in L_interpret(results, sys_names):
        L.append(s)
    L.append("")
    L.append("## Caveat(诚实声明)\n")
    L.append("- **样本限制**：各线题数见表；单题结果会改变百分比。这是本批条件下的描述，不单独证明题库难度、机制必要性或泛化区分度。")
    L.append("- **拒答合同**:新题按 abstention_kind 区分从未记录、已停止统计与范围外；旧题仅在显式研究模式保留宽松合同。")
    if meta.get("judge_mode") == "semantic":
        L.append("- **语义判分**：所有答案由LLM结合公开材料与已审参考判断；格式、附带事实和语义主分分别记录。当前为研究验证路径，模型判断仍需校准。")
    else:
        L.append("- **旧判分路径**：包含确定性值匹配和有限LLM回落；本报告不把旧规则通过等同于独立语义确认。")
    trunc = meta.get("fullctx_truncated")
    if trunc is not None:
        L.append(f"- **FullContext 截断**:全语料 {meta['corpus_chars']} 字符,预算 {FULLCTX_CHAR_BUDGET} → "
                 + ("**已截断**(部分靠后周期的文档未进入回答上下文)。"
                    if trunc else "**未截断**(完整语料已送入回答上下文，不代表模型完整利用了它)。"))
    L.append("- **单链 / ONE world**:一次 ingest 问全部题,无跨链泛化检验。")
    L.append("- **可插拔架构**:已支持 adapter 模式接入外部记忆系统(mem0/zep/memOS 等)。")

    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[报告] 已写入 {out_path}")


def L_interpret(results: dict, sys_names: list) -> list:
    """基于实测数字生成解读(逐线最强/最弱系统 + 区分度结论)。"""
    def la(s, ln):
        d = results[s]["agg"]["by_line"].get(ln)
        return d["acc"] if d and d["acc"] is not None else None

    ds = discrimination_summary(results, sys_names)
    if ds["status"] == "incomplete":
        return ["存在未计分记录：执行故障、不可判分及未决记录不进入能力分母；复核前不比较系统排名。"]
    if ds["status"] == "insufficient_systems":
        return ["至少需要两个不同系统才能比较跨系统分数。本次成绩只描述参评系统在已计分题目上的表现。"]
    out = []
    rank_str = " / ".join(f"{s}={ds['overall'][s]:.0%}" for s in ds["ranking"])
    out.append(f"1. **系统分数（降序）** {rank_str}；最高分 {ds['best']:.0%}，距满分 {ds['headroom']:.0%}。")
    # 最强区分线
    valid = {ln: v for ln, v in ds["spreads"].items() if v is not None}
    if valid:
        top_ln = max(valid, key=valid.get)
        accs = {s: la(s, top_ln) for s in sys_names}
        accs = {s: a for s, a in accs.items() if a is not None}
        if accs:
            hi = max(accs, key=accs.get)
            lo = min(accs, key=accs.get)
            out.append(f"2. **本批逐线最大分差** {top_ln.split('_')[0]}(离差 {valid[top_ln]:.0%}):"
                       f"{hi}={accs[hi]:.0%}，{lo}={accs[lo]:.0%}。")
    out.append("3. **解释范围**：这里只比较本批分数差；题库难度、机制必要性及泛化区分度仍需独立实验。")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def _load_questions(bench_path: Path) -> list:
    """读题库;兼容 list 与 {questions:[...]} 两种格式。"""
    data = json.loads(Path(bench_path).read_text(encoding="utf-8-sig"))
    return data if isinstance(data, list) else data.get("questions", data)


def main():
    ap = argparse.ArgumentParser(description="多系统记忆评测(七线判分 + 区分度)")
    ap.add_argument("--bench", default=str(DEFAULT_BENCH), help="06_grounded_questions.json")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="05_corpus.json")
    ap.add_argument("--about", default="", help="00_about.json(答题协议);默认取 bench 同目录")
    ap.add_argument("--allow-unverified", action="store_true", help="仅历史研究：允许无发布资格的输入，成绩不得作为正式结果")
    ap.add_argument("--no-protocol", action="store_true", help="不注入答题协议(消融对照)")
    ap.add_argument("--judge-mode", choices=("legacy", "semantic"), default=None,
                    help="Defaults to semantic for a published A policy, otherwise legacy; semantic is research-only")
    ap.add_argument("--scoring-policy", help="Explicit policy version; otherwise inferred from the exact public policy block")
    ap.add_argument("--grading-method", default="direct/v1", help="Semantic grading method; default direct/v1")
    ap.add_argument("--semantic-review", type=Path, help="Input-bound report produced by tools.review_semantics")
    ap.add_argument("--judge-model", help="Explicit semantic judge model; defaults to configured JUDGE_MODEL")
    ap.add_argument("--max-judge-calls", type=int, help="Semantic physical-call budget; defaults to questions × systems × calls_per_answer")
    ap.add_argument("--systems", default="A,B,C",
                    help="逗号分隔,如 A,B,C 或 simpleMem,mem0,zep,memos")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--output-dir", type=Path, help="Owned calibration directory when invoked by pipeline.factory")
    ap.add_argument("--smoke", action="store_true", help="烟测:每 line 取 1 题快速验通管线")
    ap.add_argument("--filter-easy", action="store_true", help="评测后导出 filtered/，默认剔除全员答对题")
    ap.add_argument("--keep-easy-ratio", type=float, default=None, help="全员答对题保留比例(0..1)，同时启用筛选")
    ap.add_argument("--filter-seed", type=int, default=0, help="简单题抽样种子")
    ap.add_argument("--preserve-capability", action="append", default=[], help="筛选时保留该能力全部题，供判分复核；可重复指定")
    args = ap.parse_args()

    sys_names = [s.strip() for s in args.systems.split(",") if s.strip()]
    sys_names = [s.upper() if s.upper() in ("A", "B", "C") else s for s in sys_names]
    filter_enabled = args.filter_easy or args.keep_easy_ratio is not None or bool(args.preserve_capability)
    keep_easy_ratio = args.keep_easy_ratio if args.keep_easy_ratio is not None else 0.0
    if filter_enabled:
        try:
            validate_filter_options(sys_names, keep_easy_ratio, args.filter_seed)
        except ValueError as exc:
            ap.error(str(exc))

    all_q = _load_questions(args.bench)
    from pipeline.quality import require_release
    release = require_release(args.bench, allow_unverified=args.allow_unverified, corpus_path=args.corpus)
    # Inspect the published protocol even for an ablation request: disabling it
    # must not silently route an A-policy benchmark through legacy fast paths.
    about_path = Path(args.about) if args.about else Path(args.bench).parent / "00_about.json"
    published_protocol = load_protocol(about_path)
    try:
        published_policy = protocol_scoring_policy(published_protocol)
        source_about = Path(args.bench).parent / "00_about.json"
        if args.about and source_about.resolve() != about_path.resolve():
            source_policy = protocol_scoring_policy(load_protocol(source_about))
            if source_policy is not None and published_policy != source_policy:
                raise ValueError("--about cannot remove the benchmark's published scoring policy")
        if args.scoring_policy is not None and args.scoring_policy != published_policy:
            raise ValueError("--scoring-policy does not match the exact published policy block")
        args.scoring_policy = published_policy
        if any(requires_semantic_grading(q) for q in all_q) and published_policy is None:
            raise ValueError("Process questions require their published A policy; they cannot be silently excluded by legacy selection")
        args.judge_mode = args.judge_mode or ("semantic" if published_policy else "legacy")
        if published_policy and (args.judge_mode != "semantic" or args.no_protocol):
            raise ValueError("A-policy benchmarks require semantic grading and the published protocol")
        from eval.semantic_judge import resolve_scoring_config
        scoring = resolve_scoring_config(args.scoring_policy, args.grading_method)
        if published_policy and args.semantic_review is None:
            from pipeline.grounding_review import REVIEW_ARTIFACT
            candidate_review = Path(args.bench).parent / REVIEW_ARTIFACT
            if candidate_review.is_file():
                args.semantic_review = candidate_review
        if args.judge_mode == "semantic" and args.semantic_review is None:
            raise ValueError("Semantic grading requires --semantic-review before memory ingestion")
        if args.judge_mode == "legacy" and (args.semantic_review or args.judge_model or args.max_judge_calls is not None):
            raise ValueError("Semantic review/model/budget options require --judge-mode semantic")
    except ValueError as exc:
        ap.error(str(exc))
    protocol = "" if args.no_protocol else published_protocol
    proto_on = bool(protocol)
    policy_metadata = {"scoring_configuration": scoring} if published_policy else {}
    questions = ([q for q in all_q if isinstance(q.get("question"), str) and q["question"].strip()]
                 if args.judge_mode == "semantic" else [q for q in all_q if is_judgeable(q)])
    n_unjudge = len(all_q) - len(questions)
    n_eligible = len(questions)

    if args.smoke:
        by_ln = {}
        for q in questions:
            by_ln.setdefault(q.get("line", "semantic"), q)
        questions = list(by_ln.values())
        print(f"[SMOKE] 烟测:{len(questions)} 题(每线 1)× 系统 {sys_names}")

    bid = qa_cache.bench_id(questions)   # QA 断点续传键(题面集合 hash;题变即换键)

    probe = _EvalProbe(output_dir=args.output_dir) if args.output_dir else _EvalProbe()
    probe.start(bench=args.bench, corpus=str(args.corpus), model=config.MODEL,
                systems=sys_names, n_total=len(all_q), n_judgeable=n_eligible,
                n_eligible=n_eligible, n_selected=len(questions), n_excluded=n_unjudge,
                count_semantics={"n_judgeable": "legacy_alias_for_n_eligible_not_scored"},
                protocol=proto_on, release=release, **policy_metadata)

    docs = load_corpus(args.corpus)
    solver_sources = [*(ROOT / "eval" / "memory_systems").glob("*.py"),
                      ROOT / "eval" / "memory_interface.py", ROOT / "eval" / "baseline_r1.py", ROOT / "config.py"]
    evaluation_context = make_evaluation_context(docs, protocol)
    semantic_judge = None
    if args.judge_mode == "semantic":
        from eval.semantic_judge import SemanticJudge
        from llm_trace import redact
        reference_report = json.loads(args.semantic_review.read_text(encoding="utf-8"))
        if (reference_report.get("binding", {}).get("visible_view") or {}).get("include_titles"):
            ap.error("Current solver omits titles; semantic reference must use the same view")
        def record_judgement(event):
            path = probe.run_dir / "semantic_judgements.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(redact(event, config._trace_secrets()), ensure_ascii=False) + "\n")
        semantic_judge = SemanticJudge(reference_report, questions,
            json.loads(Path(args.corpus).read_text(encoding="utf-8-sig")), protocol,
            model=args.judge_model or config.JUDGE_MODEL,
            chat_json=lambda step, messages, **kw: config.chat_json(messages, **kw),
            max_calls=args.max_judge_calls if args.max_judge_calls is not None else
                len(questions) * len(sys_names) * scoring["calls_per_answer"],
            record=record_judgement, scoring_policy=args.scoring_policy, grading_method=args.grading_method)
    cache_context = {"evaluation_context": evaluation_context,
                     "solver": {"corpus_hash": qa_cache.digest(docs),
                         "retrieval_source_hash": qa_cache.digest({str(p.relative_to(ROOT)): p.read_text(encoding="utf-8")
                             for p in solver_sources if p.is_file()}),
                         "protocol_enabled": proto_on}}
    corpus_chars = sum(len(header(s, d)) + len(c) for s, d, c in docs)

    import collections as _c
    by_line_n = _c.Counter(q.get("line", "semantic") for q in questions)
    print(f"[multi_system] bench={args.bench}")
    print(f"[multi_system] 本次选取 {len(questions)}/{len(all_q)} 题"
          + (f"(输入预排除 {n_unjudge})" if n_unjudge else "")
          + " | 逐线: " + " ".join(f"{ln.split('_')[0]}={by_line_n[ln]}" for ln in LINE_ORDER if ln in by_line_n))
    print(f"[multi_system] 语料 {len(docs)} docs / {corpus_chars} 字符 | 系统={sys_names} | "
          f"协议注入={'是' if proto_on else '否'}")

    # ── 构建记忆系统实例(工厂)──
    from eval.memory_systems import make_system

    # 语料 → session 列表(adapter 接口要求)
    sess_map = {}
    for sid, date, content in docs:
        if sid not in sess_map:
            sess_map[sid] = {"session_id": sid, "date": date, "docs": []}
        sess_map[sid]["docs"].append(content)
    sessions = [sess_map[k] for k in sorted(sess_map)]

    trace_path = probe.run_dir / "llm_attempts.jsonl"
    with trace_scope(trace_path, "eval.prepare"):
        systems, executions = prepare_systems(sys_names, sessions, make_system)

    reassessment = None
    if semantic_judge is not None:
        review_hash = semantic_judge.cache_context["reference_review_hash"]
        reassessment = ReassessmentTracker(evaluation_context, review_hash,
            path=closure_path(qa_cache.CACHE_DIR, evaluation_context, review_hash))
    results = {}
    system_elapsed = {}
    def finish_system(name):
        if "agg" not in results[name]:
            results[name]["agg"] = aggregate(results[name]["records"])
        agg = results[name]["agg"]
        probe.sys_done(name, agg, system_elapsed[name])
        per_line = " ".join(f"{ln.split('_')[0]}={(agg['by_line'].get(ln) or {}).get('acc')}"
                            for ln in LINE_ORDER if ln in agg["by_line"])
        print(f"[{name}] 完成 ({system_elapsed[name]:.1f}s) overall={agg['overall']['acc']} | {per_line}")
    fullctx_truncated = None
    for s in sys_names:
        t = time.time()
        recs = run_system(s, questions, systems[s], workers=args.workers,
                          protocol=protocol, probe=probe, bench_id=bid, cache_context=cache_context,
                          execution=executions[s], trace_path=trace_path, judge_fn=semantic_judge,
                          reassessment=reassessment)
        system_elapsed[s] = time.time() - t
        results[s] = {"records": recs, "execution": executions[s]}
        if s == "B":
            fc = next((r.get("_fullctx") for r in recs if r.get("_fullctx")), None)
            fullctx_truncated = fc["truncated"] if fc else None
        if reassessment is None:
            finish_system(s)
    reassessment_closure = (close_semantic_reassessment(results, reassessment) if reassessment is not None else None)
    if reassessment is not None:
        for s in sys_names:
            finish_system(s)

    # 打印表
    print_table(results, sys_names)

    # 落盘 JSON(落到本次 eval 独立目录)
    out_json = probe.run_dir / "results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    counts = selection_metadata(len(all_q), n_eligible, len(questions), results)
    dump = {
        "bench": args.bench, "corpus": args.corpus, "model": config.MODEL,
        "release": release, "judge_version": semantic_judge.cache_context["version"] if semantic_judge else JUDGE_VERSION,
        "judge_mode": args.judge_mode,
        **policy_metadata,
        "grading_context": {**semantic_judge.cache_context, "reassessment_policy": REASSESSMENT_VERSION}
                           if semantic_judge else {"model": getattr(config, "JUDGE_MODEL", config.MODEL)},
        "reassessment_closure": reassessment_closure,
        "evaluation_context": evaluation_context,
        "result_scope": "research_only" if release.get("override") or semantic_judge else "release_eligible",
        "systems": sys_names, "top_k": TOP_K, "fullctx_char_budget": FULLCTX_CHAR_BUDGET,
        "fullctx_truncated": fullctx_truncated, "protocol_injected": proto_on,
        **counts,
        "results": {s: {"agg": results[s]["agg"],
                        "execution": results[s]["execution"],
                        "evaluation_context": evaluation_context,
                        "records": results[s]["records"]} for s in sys_names},
        "discrimination": discrimination_summary(results, sys_names),
    }
    out_json.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[结果] 已写入 {out_json}")

    # 报告
    meta = {
        "bench": args.bench, "corpus": args.corpus,
        "judge_version": dump["judge_version"], "judge_mode": args.judge_mode,
        "judge_model": semantic_judge.model if semantic_judge else getattr(config, "JUDGE_MODEL", config.MODEL),
        "release": release,
        **policy_metadata,
        **counts,
        "n_sessions": len({s for s, _, _ in docs}),
        "n_docs": len(docs), "corpus_chars": corpus_chars,
        "fullctx_truncated": fullctx_truncated, "protocol_injected": proto_on,
    }
    write_report(results, sys_names, meta, out_path=probe.run_dir / "report.md")
    if filter_enabled:
        filter_report = export_filtered_benchmark(
            Path(args.bench), {s: results[s]["records"] for s in sys_names},
            probe.run_dir / "filtered", keep_easy_ratio=keep_easy_ratio, seed=args.filter_seed,
            preserve_capabilities=args.preserve_capability,
            corpus=Path(args.corpus), about=about_path if about_path.is_file() else None,
            result_paths=[out_json], allow_unverified=args.allow_unverified,
            protocol_enabled=not args.no_protocol)
        fc = filter_report["counts"]
        print(f"[筛题] 全员答对 {fc['all_correct']}，剔除 {fc['removed_easy']}，"
              f"保留 {fc['kept']} → {probe.run_dir / 'filtered'}")
    probe.done(disc=discrimination_summary(results, sys_names))


if __name__ == "__main__":
    main()
