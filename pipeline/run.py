"""
pipeline.run —— Run / Stage 通用基建(域无关)。详见 docs/anchors/run_system_design.md。

一次"造一个 benchmark" = 一个 Run(一个 output/runs/<run_id>/ 目录,自带 manifest/run.log/prompts.jsonl)。
pipeline = 一串命名 Stage;driver 按 --from/--to/--only 跑区间,幂等跳过已完成(manifest 为准)。

这里【只放基建】,不含任何工厂/域知识;具体 stage(whitepaper/world/orders/…)在 pipeline/factory.py 注册。
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Callable
from contextlib import contextmanager
from copy import deepcopy
from llm_trace import trace_scope, redact, failure_record
import errno, json, os, time, threading, sys

if os.name == "nt":
    import msvcrt
else:
    import fcntl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                       # 保证 config 可导入(Tracer 用)
RUNS_DIR = ROOT / "output" / "runs"


def _atomic_write_json(path: Path, obj) -> None:
    """在目标目录写临时文件后原子替换，避免进程中断留下半截 JSON。"""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # Windows scanners/readers can briefly hold a destination without
        # sharing delete access.  The complete temporary file is safe to retry;
        # keep the wait short and bounded so a real permission problem still
        # surfaces instead of stalling a generation run.
        for attempt in range(8):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.05 * (2 ** attempt), 0.5))
    finally:
        tmp.unlink(missing_ok=True)


def _jsonl_record_count(path: Path) -> int:
    """以已经落盘的非空 JSONL 行为续跑调用计数的唯一基线。"""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def new_run_id(scenario: str) -> str:
    """run_id = <scenario>__<时间戳>,唯一且可排序。"""
    return f"{scenario}__{time.strftime('%Y%m%d-%H%M%S')}"


def _env_snapshot() -> dict:
    """run 启动时快照【工程配置】(模型/并发/端点)进 manifest。让监控显示这一轮【真实跑的】配置,
    而非读 config 此刻的 env(历史 run / 中途改过 env 都会失真)。★api_key 绝不入快照。"""
    try:
        import config
        return {"model": config.MODEL,
                "discriminator_model": config.DISCRIMINATOR_MODEL,
                "structure_model": config.STRUCTURE_MODEL,
                "reviewer_model": config.REVIEWER_MODEL,
                "judge_model": config.JUDGE_MODEL,
                "llm_concurrency": config.LLM_CONCURRENCY,
                "reasoning_effort": config.REASONING_EFFORT,
                "base_url": config.BASE_URL or ""}
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════════════════════
# Tracer:每次 LLM 调用记到 <run>/prompts.jsonl(线程安全;网络在锁外 → 真并发)
# ════════════════════════════════════════════════════════════════════════════
class Tracer:
    def __init__(self, run: "Run"):
        self.pfile = run.dir / "prompts.jsonl"
        self.n = _jsonl_record_count(self.pfile)
        self._lock = threading.Lock()

    def chat_json(self, step, messages, **kw):
        import config
        t0 = time.time()
        try:
            with trace_scope(self.pfile.with_name("llm_attempts.jsonl"), step):
                out = config.chat_json(messages, **kw)    # 网络调用在锁外 → 真并发
            ok = not (isinstance(out, dict) and "__error__" in out)
        except Exception as e:
            out, ok = failure_record(e, secrets=tuple(value for key, value in os.environ.items()
                if key.endswith(("API_KEY", "API_TOKEN", "ACCESS_TOKEN", "PASSWORD")))), False
        latency_ms = int((time.time() - t0) * 1000)       # ★含 config 内部 3 次重试的总耗时(慢≈逼近超时)
        with self._lock:                                  # 计数 + 写文件串行化
            self.n += 1
            self._log(self.n, step, messages, out, kw, ok, latency_ms)
        return out

    def chat_text(self, step, messages, **kw):
        """记录一次只要求自然语言正文的调用；异常仍作为显式失败返回。"""
        import config
        t0 = time.time()
        try:
            with trace_scope(self.pfile.with_name("llm_attempts.jsonl"), step):
                out = config.chat(messages, **kw)
            ok = isinstance(out, str) and bool(out.strip())
        except Exception as e:
            out, ok = failure_record(e, secrets=tuple(value for key, value in os.environ.items()
                if key.endswith(("API_KEY", "API_TOKEN", "ACCESS_TOKEN", "PASSWORD")))), False
        latency_ms = int((time.time() - t0) * 1000)
        with self._lock:
            self.n += 1
            self._log(self.n, step, messages, out, kw, ok, latency_ms)
        return out

    def _log(self, i, step, messages, out, kw, ok=True, latency_ms=0):
        rec = {"i": i, "ts": time.time(), "latency_ms": latency_ms, "ok": ok, "step": step,
               "trace_version": 2, "messages": messages, "output": out,
               "system": next((m["content"] for m in messages if m["role"] == "system"), ""),
               "user": next((m["content"] for m in messages if m["role"] == "user"), ""),
               "params": {k: v for k, v in kw.items()
                          if k in ("temperature", "max_tokens", "model", "retries", "strict_json")},
               "out_preview": json.dumps(out, ensure_ascii=False)[:600] if isinstance(out, (dict, list)) else str(out)[:600]}
        secrets = tuple(value for key, value in os.environ.items()
                        if key.endswith(("API_KEY", "API_TOKEN", "ACCESS_TOKEN", "PASSWORD")))
        with self.pfile.open("a", encoding="utf-8") as f:
            f.write(json.dumps(redact(rec, secrets), ensure_ascii=False) + "\n")


# ════════════════════════════════════════════════════════════════════════════
# Stage:一道工序。fn(run) 从 run 读 needs 的产物、写本步产物。
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Stage:
    name: str
    needs: list            # 依赖的前序 stage 名(driver 据此校验能否从此起跑)
    fn: Callable           # fn(run) -> None
    artifact: str          # 产物文件名(NN_<name>.json)
    is_current: Callable | None = None  # Optional content/version freshness check.
    refreshes: tuple[str, ...] = ()  # Stale ancestors this stage explicitly revalidates before publishing.
    freshness_covers: tuple[str, ...] = ()  # Ancestors fully replayed by this stage's freshness check.


# ════════════════════════════════════════════════════════════════════════════
# Run:一次造 benchmark 的全部状态(目录 + manifest + logger + tracer + 产物读写)
# ════════════════════════════════════════════════════════════════════════════
class Run:
    def __init__(self, scenario: str, run_id: str, tag=None, config_meta: dict | None = None):
        self.scenario = scenario
        self.run_id = run_id
        self.tag = tag
        self.dir = RUNS_DIR / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._manifest_lock = threading.RLock()
        self.log = self._make_logger()
        self.tracer = Tracer(self)
        # 初始化同样会改 manifest，必须与 stage 共用进程锁；否则第二个进程可能在
        # 真正进入 stage 前，就先用旧快照覆盖正在运行的进程。
        with self.stage_write_lock("initialize"):
            self.manifest = self._load_or_init_manifest(tag, config_meta or {})
            self.manifest["env"] = _env_snapshot()  # 工程配置快照(模型/并发/端点),供监控集中展示
            self._save_manifest()

    # ── 产物读写(stage 只跟 Run 打交道,不碰路径)──
    def write(self, artifact: str, obj):
        _atomic_write_json(self.dir / artifact, obj)

    def read(self, artifact: str):
        return json.loads((self.dir / artifact).read_text(encoding="utf-8"))

    def has(self, artifact: str) -> bool:
        return (self.dir / artifact).exists()

    @contextmanager
    def stage_write_lock(self, stage_name: str):
        """以非阻塞进程锁保证同一 Run 同时只有一个 stage 写入。"""
        lock_file = (self.dir / ".run.lock").open("a+b")
        acquired = False
        try:
            try:
                if os.name == "nt":
                    lock_file.seek(0, os.SEEK_END)
                    if lock_file.tell() == 0:
                        lock_file.write(b"\0")
                        lock_file.flush()
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                raise RuntimeError(
                    f"Run {self.run_id} 正由另一进程写入，拒绝并发执行 stage {stage_name}"
                ) from exc
            yield
        finally:
            if acquired:
                if os.name == "nt":
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    # ── manifest(单一真相)──
    def _load_or_init_manifest(self, tag, config_meta) -> dict:
        mf = self.dir / "manifest.json"
        if mf.exists():
            m = json.loads(mf.read_text(encoding="utf-8"))
            # 区间参数的 None 也是本次调用的真实选择，必须覆盖上一次残留的
            # --only/--from/--to；其余可选项仍只在显式提供时更新。
            for key, value in config_meta.items():
                if value is not None or key in {"from", "to", "only"}:
                    m["config"][key] = value
            if tag is not None:
                m["tag"] = tag
        else:
            m = {"run_id": self.run_id, "scenario": self.scenario, "tag": tag,
                 "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "running",
                 "config": config_meta, "stages": {}, "algo": {}, "llm_calls": 0}
        return m

    def _reload_manifest_for_stage(self) -> None:
        """取得进程锁后刷新磁盘状态，只合并本实例真正未落盘的配置改动。"""
        mf = self.dir / "manifest.json"
        if not mf.exists():
            return
        disk = json.loads(mf.read_text(encoding="utf-8"))
        local_config = self.manifest.setdefault("config", {})
        baseline = getattr(self, "_config_snapshot", {})
        changed = {
            key: deepcopy(value) for key, value in local_config.items()
            if key not in baseline or baseline[key] != value
        }
        deleted = set(baseline) - set(local_config)
        merged_config = dict(disk.get("config") or {})
        for key in deleted:
            merged_config.pop(key, None)
        merged_config.update(changed)
        # closed_loop 会长期持有 manifest["config"] 的引用；原地更新，避免刷新后
        # 该引用变成脱离 manifest 的旧字典。
        local_config.clear()
        local_config.update(merged_config)
        disk["config"] = local_config
        self.manifest.clear()
        self.manifest.update(disk)
        self._config_snapshot = deepcopy(local_config)
        # prompts.jsonl 是调用计数真源；另一个实例可能已追加记录，旧 Tracer 也要
        # 在新 stage 开始前追平，否则会复用 i 并把 manifest.llm_calls 写小。
        if hasattr(self, "tracer"):
            with self.tracer._lock:
                self.tracer.n = _jsonl_record_count(self.tracer.pfile)

    def _save_manifest(self):
        with self._manifest_lock:
            self.manifest["llm_calls"] = self.tracer.n if hasattr(self, "tracer") else 0
            _atomic_write_json(self.dir / "manifest.json", self.manifest)
            self._config_snapshot = deepcopy(self.manifest.get("config") or {})

    def start_stage(self, name: str):
        """进入 stage 时撤销旧完成态，并记录本次尝试。"""
        with self._manifest_lock:
            state = self.manifest["stages"].setdefault(name, {})
            try:
                attempt = int(state.get("attempt", 0)) + 1
            except (TypeError, ValueError):
                attempt = 1
            for key in ("done", "ts", "finished_ts", "failed_ts", "elapsed_s",
                        "artifact", "error", "invalidated_ts", "invalidated_by"):
                state.pop(key, None)
            state.update({"status": "running", "attempt": attempt, "started_ts": time.time()})
            self.manifest["current_stage"] = name
            self.manifest["status"] = "running"
            self._save_manifest()

    def mark(self, stage_name: str, artifact: str, elapsed: float):
        with self._manifest_lock:
            state = self.manifest["stages"].setdefault(stage_name, {})
            state.update({                                                # ★merge:保留 started_ts/attempt
                "done": True, "status": "succeeded", "ts": time.strftime("%H:%M:%S"),
                "finished_ts": time.time(), "elapsed_s": round(elapsed, 1), "artifact": artifact})
            state.pop("error", None)
            self._save_manifest()

    def fail_stage(self, stage_name: str, error: BaseException, elapsed: float):
        """记录 stage 失败；旧完成标记不会在失败后复活。"""
        with self._manifest_lock:
            state = self.manifest["stages"].setdefault(stage_name, {})
            state.update({"done": False, "status": "failed", "failed_ts": time.time(),
                          "elapsed_s": round(elapsed, 1),
                          "error": f"{type(error).__name__}: {str(error)[:200]}"})
            self.manifest["status"] = "failed"
            self._save_manifest()

    def invalidate_after(self, stage_name: str, ordered_names: list[str]):
        """强制重跑上游时，将声明顺序中的全部下游 stage 标为失效。"""
        now = time.time()
        with self._manifest_lock:
            for name in ordered_names[ordered_names.index(stage_name) + 1:]:
                state = self.manifest["stages"].setdefault(name, {})
                for key in ("ts", "finished_ts", "failed_ts", "elapsed_s", "artifact",
                            "error", "started_ts"):
                    state.pop(key, None)
                state.update({"done": False, "status": "invalidated",
                              "invalidated_by": stage_name, "invalidated_ts": now})
            self._save_manifest()

    def is_done(self, stage_name: str) -> bool:
        return bool(self.manifest["stages"].get(stage_name, {}).get("done"))

    def set_algo(self, **kw):
        """记算法元数据(active_lines / entities / questions …),让 run 自描述到算法层。"""
        with self._manifest_lock:
            self.manifest.setdefault("algo", {}).update(kw)
            self._save_manifest()

    def set_status(self, status: str):
        with self._manifest_lock:
            self.manifest["status"] = status
            self._save_manifest()

    # ── logger:tee 到 stdout + <run>/run.log(带时间戳;线程安全)──
    def _make_logger(self):
        lf = self.dir / "run.log"

        def log(msg):
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            with self._lock:
                print(line, flush=True)
                with lf.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        return log


# ════════════════════════════════════════════════════════════════════════════
# _run_stage:跑一道【已注册 stage】的统一记账(start_stage→计时→mark;失败置 status)。
#   drive(幂等区间)与闭环 driver(直调绕过幂等)共用这一份,记账逻辑不再两处拷贝。
# ════════════════════════════════════════════════════════════════════════════
def _run_stage_locked(run: Run, name: str, fn, artifact: str,
                      invalidate_names: list[str] | None = None):
    """在调用方已持有 Run 写锁时执行并记账一道 stage。"""
    if invalidate_names is not None:
        run.invalidate_after(name, invalidate_names)
    t = time.time()
    run.start_stage(name)
    run.log(f"→ stage: {name}")
    try:
        fn(run)
    except BaseException as e:
        run.fail_stage(name, e, time.time() - t)
        run.log(f"✗ stage {name} 失败:{type(e).__name__}: {str(e)[:200]}")
        raise
    run.mark(name, artifact, time.time() - t)


def _run_stage(run: Run, name: str, fn, artifact: str, invalidate_names: list[str] | None = None):
    """按 drive 同款记账跑一个【已注册 stage 函数】:start_stage + 计时 + mark + 失败置 status。
    闭环 driver 用它直调 stage(绕开 drive 的幂等跳过,B3),但 stage 体本身零复制——
    diversity / ckpt 续渲 / set_algo 全归各 stage 独占,两条编排路径不再分叉。"""
    with run.stage_write_lock(name):
        # Run 对象可能早于另一进程完成上一步而创建；锁只能串行写，刷新才能避免
        # 后取得锁的旧对象把磁盘上的新 stage 状态覆盖掉。
        run._reload_manifest_for_stage()
        _run_stage_locked(run, name, fn, artifact, invalidate_names)


def _finalize_run(run: Run, stage_names: list[str] | None = None) -> str:
    """按 stage 真实状态收束 Run；失败优先于失效，禁止无关单步误报成功。"""
    with run.stage_write_lock("finalize"):
        run._reload_manifest_for_stage()
        names = stage_names or list(run.manifest.get("stages") or {})
        statuses = [run.manifest["stages"].get(name, {}).get("status") for name in names]
        if "failed" in statuses:
            status = "failed"
        elif "invalidated" in statuses:
            status = "invalidated"
        elif "running" in statuses:
            status = "running"
        else:
            status = "done"
        if status != "running":
            run.manifest["current_stage"] = ""
        run.set_status(status)
        return status


def _update_run_metadata(run: Run, *, algo: dict | None = None,
                         config_remove: tuple[str, ...] = ()) -> None:
    """安全写入 driver 级元数据，避免锁外旧快照覆盖正在推进的 stage。"""
    with run.stage_write_lock("metadata"):
        run._reload_manifest_for_stage()
        for key in config_remove:
            run.manifest.setdefault("config", {}).pop(key, None)
        if algo:
            run.manifest.setdefault("algo", {}).update(algo)
        run._save_manifest()


def _dependent_stage_names(stage_name: str, stages: list[Stage]) -> list[str]:
    """按 ``needs`` 求一个 stage 的传递后继，并保持注册顺序。

    线性位置不是依赖关系：questions 与 corpus 是两条并行分支，强制重出题只应
    失效最终 grounding，不能把完全不读 questions 的昂贵 corpus 一并作废。
    """
    affected = {stage_name}
    changed = True
    while changed:
        changed = False
        for stage in stages:
            if stage.name not in affected and any(need in affected for need in stage.needs):
                affected.add(stage.name)
                changed = True
    return [stage.name for stage in stages if stage.name in affected]


# ════════════════════════════════════════════════════════════════════════════
# driver:按 stage 序列跑 [from,to] 区间(或 only 单步);幂等跳过已完成(除非 force)
# ════════════════════════════════════════════════════════════════════════════
def drive(run: Run, stages: list, from_stage=None, to_stage=None, only=None, force=False):
    names = [s.name for s in stages]
    by_name = {s.name: s for s in stages}
    # Content-bound semantic receipts can be expensive to replay for a large
    # world.  Within one drive call, a successful freshness result remains valid
    # until any stage writes.  A write clears the whole cache, so no downstream
    # stage can rely on a result computed for an older artifact set.
    current_cache = {}
    def is_current(name):
        stage = by_name[name]
        if stage.is_current is None:
            return True
        if name not in current_cache:
            current_cache[name] = bool(stage.is_current(run))
            if current_cache[name]:
                for covered in stage.freshness_covers:
                    current_cache[covered] = True
        return current_cache[name]
    for nm in ([only] if only else names[(names.index(from_stage) if from_stage else 0):
                                          (names.index(to_stage) if to_stage else len(names) - 1) + 1]):
        st = by_name[nm]
        with run.stage_write_lock(nm):
            # 刷新、依赖判断、跳过判断和执行必须原子；否则旧 Run 实例可复活
            # 已被另一实例 invalidated 的下游。
            run._reload_manifest_for_stage()
            for need in st.needs:                          # 依赖校验:前序产物必须在
                if not run.has(by_name[need].artifact):
                    raise SystemExit(f"✗ stage『{nm}』需要前序『{need}』产物 {by_name[need].artifact},但不存在。"
                                     f"先跑 --to {need}(或 --from {need})。")
                state = run.manifest["stages"].get(need, {})
                if state and not state.get("done"):
                    raise SystemExit(f"✗ stage『{nm}』依赖的前序『{need}』"
                                     f"状态为 {state.get('status', 'incomplete')}。"
                                     f"先从 --from {need} 补跑下游。")
            # Check freshness through the dependency chain, including when this
            # stage itself is done. A saved question/corpus flag must not hide
            # an expired world opinion during a midstream restart.
            ancestors = set(st.needs)
            pending = list(st.needs)
            while pending:
                for need in by_name[pending.pop()].needs:
                    if need not in ancestors:
                        ancestors.add(need)
                        pending.append(need)
            # Descendant receipts may fully replay an ancestor (for example the
            # disclosure receipt includes the bound world opinion).  Check the
            # most downstream ancestor first so one verified receipt can cover
            # its declared inputs without weakening any standalone check.
            for need in reversed(names):
                if (need in ancestors and by_name[need].is_current is not None
                        and need not in st.refreshes
                        and not is_current(need)):
                    raise SystemExit(f"✗ stage『{nm}』依赖的前序『{need}』验收已过期。"
                                     f"先从 --from {need} 重新运行。")
            if run.is_done(nm) and not force and is_current(nm):
                run.log(f"⏭  跳过 {nm}(已完成;--force 重跑)")
                continue
            # A freshness-triggered rebuild changes upstream artifacts just as
            # an explicit force does. Its old descendants must be regenerated.
            invalidated = (_dependent_stage_names(nm, stages)
                           if force or run.is_done(nm) else None)
            _run_stage_locked(run, nm, st.fn, st.artifact, invalidated)
            current_cache.clear()
    _finalize_run(run, names)


# ════════════════════════════════════════════════════════════════════════════
# list_runs:扫 output/runs/*/manifest.json(新→旧),给 --list-runs 用
# ════════════════════════════════════════════════════════════════════════════
def list_runs() -> list:
    if not RUNS_DIR.exists():
        return []
    out = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        mf = d / "manifest.json"
        if mf.exists():
            try:
                out.append(json.loads(mf.read_text(encoding="utf-8")))
            except Exception:
                pass
    return out


def latest_run_for(scenario: str):
    for m in list_runs():
        if m.get("scenario") == scenario:
            return m["run_id"]
    return None
