"""
eval.memory_systems.amem_adapter — A-Mem (Agentic Memory) adapter。

依赖: pip install git+https://github.com/WujiangXu/A-mem-sys.git

A-Mem 基于 Zettelkasten: 每条记忆是结构化 note (keywords/context/tags/links),
LLM 自动分析并建立笔记间的图链接。检索时走 向量 + 图扩展。

Embedding 用本地 bge-small-zh-v1.5 (中文,512维,不走 API,与全系统统一底座)。
LLM 调用走 INGEST_LLM_BASE_URL（若设了），否则走 OPENAI_BASE_URL (DMXAPI)。
"""
from __future__ import annotations
import os, sys, time, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.memory_systems.base import (MemorySystem, MemoryExecutionError, execution_stage,
                                      ingest_receipt, require_response, configuration_fingerprint)
from eval.memory_systems.execution import (SDKGuard, validate_json_completion,
                                           validate_chroma_search)
from eval.multi_system import header

_env_lock = threading.Lock()


def _build_amem(model: str):
    """构造 AgenticMemorySystem，线程安全地临时切换 OPENAI_BASE_URL。"""
    from agentic_memory.memory_system import AgenticMemorySystem

    ingest_base = os.getenv("INGEST_LLM_BASE_URL")
    with _env_lock:
        prev = os.environ.get("OPENAI_BASE_URL")
        if ingest_base:
            os.environ["OPENAI_BASE_URL"] = ingest_base
        try:
            sys_obj = AgenticMemorySystem(
                model_name="BAAI/bge-small-zh-v1.5",
                llm_backend="openai",
                llm_model=model,
            )
        finally:
            if prev is not None:
                os.environ["OPENAI_BASE_URL"] = prev
            elif ingest_base:
                os.environ.pop("OPENAI_BASE_URL", None)
    return sys_obj


def _harden_llm(sys_obj):
    """A-Mem 默认 temperature=1.0 且【不传 max_tokens】(LLMController.get_completion 丢弃了该参数),
    小模型(Qwen3-8B)在结构化"记忆演化"任务上会失控生成(撞 max-model-len → 130KB 垃圾 → JSON 炸 →
    卡死)。这里把它的 LLM 调用收紧:强制 temperature=0(结构化要确定性)+ max_tokens 封顶 + 关 Qwen3
    思考模式(<think> 会在 schema 外狂输出)。换强模型后这层依然无害(只是确定性+有界)。"""
    llm = getattr(sys_obj, "llm_controller", None)
    llm = getattr(llm, "llm", None)
    client = getattr(llm, "client", None)
    model = getattr(llm, "model", None)
    if client is None or model is None:
        return   # 非 OpenAI 后端 → 跳过

    def _gc(prompt, response_format=None, temperature=0.0, max_tokens=1024):
        kw = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
            "timeout": 60,
        }
        if response_format:
            kw["response_format"] = response_format
        for attempt in range(3):
            try:
                try:
                    r = client.chat.completions.create(
                        **kw, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                except TypeError:
                    r = client.chat.completions.create(**kw)
                choice = r.choices[0]
                if getattr(choice, "finish_reason", None) == "length":
                    raise MemoryExecutionError("ingest", "llm_truncated")
                value = choice.message.content
                validate_json_completion(value, (), {"response_format": response_format}, "ingest")
                return value
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise MemoryExecutionError("ingest", "llm_failed",
                                           {"attempts": 3, "cause_type": type(e).__name__}) from e

    llm.get_completion = _gc


class AMemAdapter(MemorySystem):

    def __init__(self, top_k: int = 10, llm_model: str = None, **kwargs):
        self.top_k = top_k
        self._last_context = ""
        self._model = (llm_model
                       or os.getenv("INGEST_LLM_MODEL")
                       or os.getenv("MODEL", "gpt-4o-mini"))
        self._guard = SDKGuard()
        with execution_stage("init"):
            self._llm_endpoint_fingerprint = configuration_fingerprint(
                os.getenv("INGEST_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
            self._sys = _build_amem(self._model)
            _harden_llm(self._sys)
            self._observe_dependencies()

    def evaluation_config(self) -> dict:
        return {"configuration_status": "declared", "top_k": self.top_k,
                "llm_model": self._model, "llm_backend": "openai",
                "llm_endpoint_fingerprint": self._llm_endpoint_fingerprint,
                "embedding_model": "BAAI/bge-small-zh-v1.5",
                "llm_max_tokens": 1024, "llm_temperature": 0.0,
                "sdk_operations": "serialized", "sdk_defaults": "not_exposed"}

    def _observe_dependencies(self):
        # Pinned A-Mem catches failures in analyze_content, process_memory and
        # search_agentic. Observe below those catch blocks, then check after return.
        llm = getattr(getattr(self._sys, "llm_controller", None), "llm", None)
        if llm is not None:
            self._guard.watch(llm, "get_completion", validate_json_completion)
        retriever = getattr(self._sys, "retriever", None)
        if retriever is not None:
            self._guard.watch(retriever, "search", validate_chroma_search)
            for name in ("add_document", "delete_document"):
                self._guard.watch(retriever, name)

    def ingest_session(self, session: dict) -> dict:
        sid, date = session["session_id"], session["date"]
        n = 0
        for doc in session["docs"]:
            with execution_stage("ingest", requested_docs=len(session["docs"]), completed_docs=n):
                self._observe_dependencies()
                note_id = self._guard.run("ingest", self._sys.add_note,
                    content=header(sid, date) + doc, time=date)
                require_response(isinstance(note_id, str) and bool(note_id), "ingest")
            n += 1
        return ingest_receipt(n)

    def retrieve(self, question: str, top_k: int = None) -> str:
        self._last_context = ""
        with execution_stage("retrieve"):
            self._observe_dependencies()
            results = self._guard.run("retrieve", self._sys.search_agentic,
                                      question, k=top_k or self.top_k)
            require_response(isinstance(results, list), "retrieve")
            lines = []
            for r in results:
                require_response(isinstance(r, dict) and isinstance(r.get("content"), str), "retrieve")
                score = r.get("score", "")
                neighbor = " [linked]" if r.get("is_neighbor") else ""
                score_str = f"[score={score:.2f}]" if isinstance(score, (int, float)) else ""
                lines.append(f"{score_str}{neighbor} {r['content']}".strip())
            self._last_context = "\n".join(lines) or "(无检索结果)"
            return self._last_context

    def get_memory_snapshot(self) -> dict:
        with execution_stage("snapshot"):
            memories = self._sys.memories
            lines = [f"[{note.keywords}] {note.content}" for note in memories.values()]
            return {"text": "\n".join(lines) or "(empty)", "n_notes": len(memories)}

    def reset(self) -> None:
        with execution_stage("reset"):
            self._llm_endpoint_fingerprint = configuration_fingerprint(
                os.getenv("INGEST_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
            self._sys = _build_amem(self._model)
            _harden_llm(self._sys)
            self._observe_dependencies()
            self._last_context = ""
