"""
eval.memory_systems.memos_adapter — MemOS (MemTensor) 记忆系统 adapter。

纯 HTTP API,无额外 pip 依赖(只用 requests)。
环境:MEMOS_API_KEY(云)或 MEMOS_BASE_URL(自托管,默认 http://localhost:5230)。
"""
from __future__ import annotations
import os, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import requests
from eval.memory_systems.base import (MemorySystem, MemoryExecutionError, execution_stage,
                                      ingest_receipt, require_response, configuration_fingerprint)
from eval.multi_system import header


class MemOSAdapter(MemorySystem):

    def __init__(self, top_k: int = 20, **kwargs):
        self.top_k = top_k
        self.api_key = kwargs.get("api_key") or os.getenv("MEMOS_API_KEY")
        default_base = ("https://memos.memtensor.cn/api/openmem/v1" if self.api_key
                        else "http://localhost:5230/api/openmem/v1")
        self.base_url = (kwargs.get("base_url") or os.getenv("MEMOS_BASE_URL", default_base)).rstrip("/")
        self._user_id = f"eval_{uuid.uuid4().hex[:8]}"
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"
        if self.api_key:
            self._session.headers["Authorization"] = f"Token {self.api_key}"
        self._last_context = ""
        self._conv_ids: list = []
        self._pending_tasks: set = set()
        self._completion_unverified = False
        self.finalize_attempts = kwargs.get("finalize_attempts", 12)
        self.finalize_interval = kwargs.get("finalize_interval", 5.0)

    def evaluation_config(self) -> dict:
        return {"configuration_status": "declared", "top_k": self.top_k,
                "endpoint_fingerprint": configuration_fingerprint(self.base_url),
                "finalize_attempts": self.finalize_attempts, "finalize_interval": self.finalize_interval,
                "include_preference": True, "external_configuration": "not_exposed"}

    def _conv_id(self, date: str) -> str:
        return date.replace("-", "")

    def _post(self, endpoint, payload, stage, timeout=30):
        with execution_stage(stage):
            r = self._session.post(f"{self.base_url}/{endpoint}", json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            require_response(isinstance(data, dict) and type(data.get("code")) is int, stage)
            if data["code"] != 0:
                raise MemoryExecutionError(stage, "service_rejected", {"service_code": data["code"]})
            return data

    def ingest_session(self, session: dict) -> dict:
        sid, date = session["session_id"], session["date"]
        conv_id = self._conv_id(date)
        messages = []
        for doc in session["docs"]:
            messages.extend([{"role": "user", "content": header(sid, date) + doc,
                              "chat_time": date},
                             {"role": "assistant", "content": "已记录。"}])
        if not messages:
            return ingest_receipt(0)
        with execution_stage("ingest", requested_docs=len(session["docs"]), completed_docs=0):
            data = self._post("add/message", {"user_id": self._user_id,
                "conversation_id": conv_id, "messages": messages}, "ingest", timeout=60)
            result = data.get("data")
            require_response(isinstance(result, dict), "ingest")
            if result.get("success") is False or result.get("status") == "failed":
                raise MemoryExecutionError("ingest", "service_rejected")
            status = result.get("status")
            task_id = result.get("task_id")
            completion = "completed" if status == "completed" else "accepted"
            if completion == "accepted":
                if isinstance(task_id, str) and task_id:
                    self._pending_tasks.add(task_id)
                else:
                    self._completion_unverified = True
            if conv_id not in self._conv_ids:
                self._conv_ids.append(conv_id)
            return ingest_receipt(len(session["docs"]), completion=completion)

    def finalize_ingest(self, on_progress=None) -> None:
        # Official add/message is asynchronous. code=0 is acceptance, not completion.
        # https://memos-docs.openmem.net/cn/api_docs/message/get_status/
        if self._completion_unverified:
            raise MemoryExecutionError("finalize", "completion_unverified",
                                       {"reason": "accepted_without_task_status"})
        total = len(self._pending_tasks)
        for attempt in range(self.finalize_attempts):
            for task_id in list(self._pending_tasks):
                data = self._post("get/status", {"task_id": task_id}, "finalize").get("data")
                require_response(isinstance(data, dict) and data.get("task_id") == task_id
                                 and data.get("status") in ("running", "completed", "failed"),
                                 "finalize")
                if data["status"] == "failed":
                    raise MemoryExecutionError("finalize", "processing_failed")
                if data["status"] == "completed":
                    self._pending_tasks.remove(task_id)
            if on_progress:
                on_progress(total - len(self._pending_tasks), total)
            if not self._pending_tasks:
                return
            if attempt + 1 < self.finalize_attempts:
                time.sleep(self.finalize_interval)
        raise MemoryExecutionError("finalize", "completion_timeout",
                                   {"pending_tasks": len(self._pending_tasks)})

    def retrieve(self, question: str, top_k: int = None) -> str:
        self._last_context = ""
        k = top_k or self.top_k
        with execution_stage("retrieve"):
            data = self._post("search/memory", {"user_id": self._user_id,
                "query": question, "memory_limit_number": min(k, 25),
                "include_preference": True, "preference_limit_number": min(k, 25)}, "retrieve")
            d = data.get("data")
            require_response(isinstance(d, dict) and
                             any(key in d for key in ("memory_detail_list", "preference_detail_list")),
                             "retrieve")
            lines = []
            for key in ("memory_detail_list", "preference_detail_list"):
                rows = d.get(key, [])
                require_response(isinstance(rows, list), "retrieve")
                for row in rows:
                    require_response(isinstance(row, dict), "retrieve")
                    if key == "memory_detail_list":
                        val, name = row.get("memory_value", ""), row.get("memory_key", "")
                        require_response(isinstance(val, str) and isinstance(name, str)
                                         and bool(val or name), "retrieve")
                        text = f"{name}: {val}" if name and val else val or name
                        score = row.get("relativity", 0)
                        lines.append(f"[score={score:.2f}] {text.strip()}")
                    else:
                        pref = row.get("preference")
                        require_response(isinstance(pref, str) and bool(pref.strip()), "retrieve")
                        lines.append(f"[pref] {pref.strip()}")
            self._last_context = "\n".join(lines) or "(无检索结果)"
            return self._last_context

    def get_memory_snapshot(self) -> dict:
        with execution_stage("snapshot"):
            all_msgs = []
            for cid in self._conv_ids:
                data = self._post("get/message", {"user_id": self._user_id,
                    "conversation_id": cid}, "snapshot", timeout=15).get("data")
                require_response(isinstance(data, dict) and
                                 isinstance(data.get("message_detail_list"), list), "snapshot")
                all_msgs.extend(data["message_detail_list"])
            text = "\n".join(m["content"] for m in all_msgs)
            return {"text": text or "(empty)", "n_messages": len(all_msgs)}

    def reset(self) -> None:
        for cid in self._conv_ids:
            self._post("delete/message", {"user_id": self._user_id,
                "conversation_id": cid}, "reset", timeout=10)
        self._conv_ids = []
        self._pending_tasks = set()
        self._completion_unverified = False
        self._user_id = f"eval_{uuid.uuid4().hex[:8]}"
        self._last_context = ""
