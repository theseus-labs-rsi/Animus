"""
eval.memory_systems.iterative — 迭代检索 baseline(= 原系统 C)。

两跳:retrieve → LLM 抽桥实体 → 用桥再 retrieve → 合并片段。
L2 多跳题真正需要的架构。
"""
from __future__ import annotations
import sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import config
from eval.memory_systems.simplemem import SimpleMem
from eval.memory_interface import EmbedMemory
from eval.memory_systems.base import execution_stage, require_response

HOP1_SYSTEM = """你在做多跳检索的【第一跳】。给定一个最终问题和一批检索片段,\
该问题需要先定位一个【中间实体】(桥),才能继续查到最终答案。

【任务】只根据片段,找出回答最终问题所必需的那个中间实体(通常是人名/部门/职位)。
【输出严格 JSON】{"bridge": "中间实体值", "subquestion": "用该中间实体改写出的、用于第二跳检索的子问题"}
若片段里找不到中间实体,bridge 填空串。"""


class Iterative(SimpleMem):
    """继承 SimpleMem 的 ingest/finalize(共享 EmbedMemory),覆写 retrieve 做两跳。"""

    def __init__(self, top_k: int = 3, embed_mem: EmbedMemory = None,
                 max_tokens: int = 2048):
        super().__init__(top_k=top_k, embed_mem=embed_mem)
        self.max_tokens = max_tokens
        self._retrieval_state = threading.local()

    def evaluation_config(self) -> dict:
        return {**super().evaluation_config(), "max_tokens": self.max_tokens,
                "bridge_model": config.MODEL, "bridge_temperature": 0.0}

    def retrieve(self, question: str, top_k: int = None) -> str:
        self._retrieval_state.context, self._retrieval_state.diagnostics = "", {}
        with execution_stage("retrieve"):
            return self._retrieve(question, top_k)

    def _retrieve(self, question: str, top_k: int = None) -> str:
        k = top_k or self.top_k
        snip1 = self._mem.retrieve(question, top_k=k)
        ctx1 = "\n\n".join(f"[片段{i+1}] {s}" for i, s in enumerate(snip1)) or "(无检索结果)"

        bridge, subq = "", ""
        msgs = [
            {"role": "system", "content": HOP1_SYSTEM},
            {"role": "user", "content": (
                f"【检索片段】\n{ctx1}\n\n【最终问题】{question}\n\n严格 JSON:"
            )},
        ]
        data = config.chat_json(msgs, temperature=0.0, max_tokens=self.max_tokens)
        require_response(isinstance(data, dict) and isinstance(data.get("bridge"), str)
                         and isinstance(data.get("subquestion"), str),
                         "retrieve", component="bridge_completion")
        bridge = data["bridge"].strip()
        subq = data["subquestion"].strip()

        hop2_query = subq or (f"{bridge} {question}" if bridge else question)
        snip2 = self._mem.retrieve(hop2_query, top_k=k)

        merged = list(dict.fromkeys(snip1 + snip2))
        if not merged:
            ctx = "(无检索结果)"
        else:
            ctx = "\n\n".join(f"[片段{i+1}] {s}" for i, s in enumerate(merged))

        self._retrieval_state.context = ctx
        self._retrieval_state.diagnostics = {"bridge": bridge, "subquestion": subq,
                           "n_snip1": len(snip1), "n_snip2": len(snip2),
                           "n_merged": len(merged)}
        return ctx

    def get_diagnostics(self) -> dict:
        return dict(getattr(self._retrieval_state, "diagnostics", {}))

    def get_retrieved_context(self) -> str:
        return getattr(self._retrieval_state, "context", "")

    def reset(self) -> None:
        super().reset()
        self._retrieval_state = threading.local()
