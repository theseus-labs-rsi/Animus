"""
eval.memory_systems.zep_adapter — Zep 时序知识图 adapter。

ZEP_API_PROFILE=ce_0_27_2 显式绑定已核实的 CE 协议，可验证消息索引完成。
默认 auto_legacy 依有无 ZEP_API_KEY 选择 legacy Cloud / CE；接受写入不等于完成。
未知部署版本不自动冒充 v0.27.2。ZEP_BASE_URL 默认为 http://localhost:8998。
"""
from __future__ import annotations
import hashlib, json, os, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import requests as _req
from eval.memory_systems.base import (MemorySystem, MemoryExecutionError, execution_stage,
                                      ingest_receipt, require_response, configuration_fingerprint)
from eval.multi_system import header

CHUNK_LIMIT = 2400


def _split_text(text: str, limit: int = CHUNK_LIMIT) -> list:
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit:
            if cur.strip():
                chunks.append(cur.strip())
            cur = line
        else:
            cur += "\n" + line if cur else line
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or [text[:limit]]


class ZepAdapter(MemorySystem):
    """Zep CE / legacy Cloud transport with explicit acceptance and readiness."""

    def __init__(self, top_k: int = 10, ingest_wait: float = 2.0, **kwargs):
        self.top_k, self.ingest_wait = top_k, ingest_wait
        self._last_context = ""
        self._has_ingested = False
        self.api_profile = kwargs.get("api_profile") or os.getenv("ZEP_API_PROFILE", "auto_legacy")
        self.finalize_attempts = kwargs.get("finalize_attempts", 12)
        self.finalize_interval = kwargs.get("finalize_interval", 5.0)
        self._ingest_batch = uuid.uuid4().hex
        self._expected_messages = {}
        with execution_stage("init"):
            if self.api_profile not in {"auto_legacy", "ce_legacy", "cloud_legacy", "ce_0_27_2"}:
                raise MemoryExecutionError("init", "unsupported_api_profile")
            require_response(type(self.finalize_attempts) is int and self.finalize_attempts > 0
                             and isinstance(self.finalize_interval, (int, float))
                             and self.finalize_interval >= 0, "init", component="poll_configuration")
            cloud_key = os.getenv("ZEP_API_KEY")
            self._mode = ("cloud" if self.api_profile == "cloud_legacy" or
                          (self.api_profile == "auto_legacy" and cloud_key) else "ce")
            if self._mode == "cloud":
                require_response(bool(cloud_key), "init", component="cloud_credentials")
                self._init_cloud(cloud_key)
            else:
                self._init_ce()

    def evaluation_config(self) -> dict:
        return {"configuration_status": "declared", "api_profile": self.api_profile,
                "mode": self._mode, "top_k": self.top_k, "chunk_limit": CHUNK_LIMIT,
                "ingest_wait": self.ingest_wait, "finalize_attempts": self.finalize_attempts,
                "finalize_interval": self.finalize_interval,
                "endpoint_fingerprint": configuration_fingerprint(getattr(self, "_base", None)),
                "external_configuration": "not_exposed"}

    def _ce_request(self, method, endpoint, stage, *, payload=None, parse=True,
                    plain_ok=False, params=None):
        with execution_stage(stage):
            kwargs = {"timeout": 30}
            if payload is not None:
                kwargs["json"] = payload
            if params is not None:
                kwargs["params"] = params
            response = getattr(self._session, method)(f"{self._base}/api/v1/{endpoint}", **kwargs)
            response.raise_for_status()
            if plain_ok:
                # v0.27.2 PostMemoryHandler writes unquoted plain text, not JSON.
                require_response(response.text.strip() == "OK", stage,
                                 component="ce_write_acknowledgement")
                return None
            if not parse or not response.content:
                return None
            return response.json()

    def _init_ce(self):
        self._base = os.getenv("ZEP_BASE_URL", "http://localhost:8998").rstrip("/")
        self._session = _req.Session()
        self._session.headers["Content-Type"] = "application/json"
        self._user_id = f"eval_{uuid.uuid4().hex[:8]}"
        self._session_id = f"sess_{uuid.uuid4().hex[:8]}"
        user_route = "user" if self.api_profile == "ce_0_27_2" else "users"
        self._ce_request("post", user_route, "init", payload={"user_id": self._user_id}, parse=False)
        self._ce_request("post", "sessions", "init", payload={
            "session_id": self._session_id, "user_id": self._user_id}, parse=False)

    def _init_cloud(self, api_key):
        from zep_cloud.client import Zep
        from zep_cloud.types import CreateUserRequest
        self.client = Zep(api_key=api_key)
        self._user_id = f"eval_{uuid.uuid4().hex[:8]}"
        self._session_id = f"sess_{uuid.uuid4().hex[:8]}"
        self.client.user.add(CreateUserRequest(user_id=self._user_id))

    def ingest_session(self, session: dict) -> dict:
        n, chunks_done = 0, 0
        for doc in session["docs"]:
            chunks = _split_text(header(session["session_id"], session["date"]) + doc)
            for chunk in chunks:
                with execution_stage("ingest", requested_docs=len(session["docs"]),
                                     completed_docs=n, accepted_chunks=chunks_done):
                    messages = [{"role_type": "user", "role": "user", "content": chunk},
                                {"role_type": "assistant", "role": "assistant", "content": "已记录。"}]
                    expected = {}
                    if self.api_profile == "ce_0_27_2":
                        for message in messages:
                            mid = uuid.uuid4().hex
                            message["metadata"] = {"eval_ingest_id": self._ingest_batch,
                                                   "eval_message_id": mid}
                            expected[mid] = {"role": message["role"],
                                "content_hash": hashlib.sha256(message["content"].encode()).hexdigest()}
                    if self._mode == "cloud":
                        from zep_cloud import Message
                        self.client.memory.add(self._session_id,
                            messages=[Message(**m) for m in messages])
                    else:
                        self._ce_request("post", f"sessions/{self._session_id}/memory",
                            "ingest", payload={"messages": messages}, parse=False,
                            plain_ok=self.api_profile == "ce_0_27_2")
                    self._expected_messages.update(expected)
                    self._has_ingested = True
                    chunks_done += 1
            n += 1
        if n and self.ingest_wait > 0:
            time.sleep(self.ingest_wait)
        return ingest_receipt(n, completion="accepted" if n else "completed",
                              accepted_chunks=chunks_done, api_profile=self.api_profile,
                              completion_scope=("message_embeddings" if self.api_profile == "ce_0_27_2"
                                                else "unverified"))

    def retrieve(self, question: str, top_k: int = None) -> str:
        self._last_context = ""
        with execution_stage("retrieve"):
            if self._mode == "cloud":
                value = self.client.memory.search(self._session_id, text=question,
                    limit=top_k or self.top_k, search_type="mmr", search_scope="facts")
                require_response(hasattr(value, "results"), "retrieve")
                rows = value.results
                # Preserve the legacy SDK's explicit optional empty field;
                # missing response/field and wrong non-null types still fail.
                if rows is None:
                    rows = []
                require_response(isinstance(rows, list), "retrieve")
                lines = []
                for row in rows:
                    text = getattr(row, "fact", None) or getattr(row, "content", None)
                    require_response(isinstance(text, str), "retrieve")
                    lines.append(text)
            else:
                versioned = self.api_profile == "ce_0_27_2"
                payload = {"text": question}
                if versioned:
                    payload["search_scope"] = "messages"
                else:
                    payload["limit"] = top_k or self.top_k
                value = self._ce_request("post", f"sessions/{self._session_id}/search", "retrieve",
                    payload=payload, params={"limit": top_k or self.top_k} if versioned else None)
                rows = value if isinstance(value, list) else (
                    value.get("results") if isinstance(value, dict) else None)
                require_response(isinstance(rows, list), "retrieve")
                lines = []
                for row in rows:
                    require_response(isinstance(row, dict), "retrieve")
                    msg = row.get("message")
                    text = msg.get("content") if isinstance(msg, dict) else row.get("content")
                    require_response(isinstance(text, str), "retrieve")
                    lines.append(text)
            self._last_context = "\n".join(lines) or "(无检索结果)"
            return self._last_context

    def _memory(self, stage):
        if self._mode == "cloud":
            with execution_stage(stage):
                return self.client.memory.get(self._session_id)
        value = self._ce_request("get", f"sessions/{self._session_id}/memory", stage)
        require_response(isinstance(value, dict), stage)
        return value

    def get_memory_snapshot(self) -> dict:
        with execution_stage("snapshot"):
            value = self._memory("snapshot")
            if isinstance(value, dict):
                facts = value.get("facts", [])
            else:
                require_response(hasattr(value, "facts"), "snapshot")
                facts = value.facts
            if facts is None:
                facts = []
            require_response(isinstance(facts, list), "snapshot")
            text = "\n".join(f["fact"] if isinstance(f, dict) else f.fact for f in facts)
            return {"text": text or "(empty)", "n_facts": len(facts)}

    def finalize_ingest(self, on_progress=None) -> dict | None:
        if not self._has_ingested:
            return
        if self.api_profile == "ce_0_27_2":
            return self._verify_message_index(on_progress)
        # This legacy API's summary cursor identifies summarized messages, not
        # completion of every embedding/extraction job. A retained unsummarized
        # tail can be perfectly valid. Neither a sleep nor cursor equality is a
        # documented whole-ingest completion signal. Check transport, then keep
        # the accepted write explicitly unverified until a versioned completion
        # contract is available; do not declare the service broken.
        self._memory("finalize")
        raise MemoryExecutionError("finalize", "completion_unverified", {
            "reason": "legacy_api_has_no_verified_whole_ingest_completion_contract",
            "mode": self._mode, "accepted": True})

    def _verify_message_index(self, on_progress=None) -> dict:
        """Verify the artifact this profile retrieves, not all background tasks.

        v0.27.2 metadata-only search joins message_embedding to message; it does
        not invoke a query embedder when text is empty. Each expected physical
        message must be returned with the same metadata ID, role and content.
        Sources: pkg/store/postgres/search_memory.go:45-68,149-162,192-217;
        pkg/store/postgres/message.go:535-556 in getzep/zep tag v0.27.2.
        The profile must be explicitly configured for the deployed API version.
        """
        expected = self._expected_messages
        if not expected:
            raise MemoryExecutionError("finalize", "completion_unverified",
                                       {"reason": "missing_expected_message_manifest"})
        query = {"search_scope": "messages", "search_type": "similarity", "text": "",
                 "metadata": {"where": {"jsonpath":
                     '$.eval_ingest_id ? (@ == ' + json.dumps(self._ingest_batch) + ')'}}}
        verified = set()
        for attempt in range(self.finalize_attempts):
            rows = self._ce_request("post", f"sessions/{self._session_id}/search", "finalize",
                                    payload=query, params={"limit": len(expected) + 1})
            require_response(isinstance(rows, list), "finalize")
            verified = set()
            for row in rows:
                message = row.get("message") if isinstance(row, dict) else None
                require_response(isinstance(message, dict), "finalize")
                metadata = message.get("metadata")
                require_response(isinstance(metadata, dict), "finalize")
                mid = metadata.get("eval_message_id")
                require_response(isinstance(mid, str) and mid in expected and mid not in verified
                                 and metadata.get("eval_ingest_id") == self._ingest_batch,
                                 "finalize", component="message_identity")
                content = message.get("content")
                require_response(isinstance(content, str) and message.get("role") == expected[mid]["role"]
                                 and hashlib.sha256(content.encode()).hexdigest() == expected[mid]["content_hash"],
                                 "finalize", component="message_content")
                verified.add(mid)
            if on_progress:
                on_progress(len(verified), len(expected))
            if verified == set(expected):
                return {"status": "ok", "completion": "completed", "scope": "message_embeddings",
                        "api_profile": self.api_profile, "profile_binding": "explicit_configuration",
                        "expected_messages": len(expected), "verified_messages": len(verified),
                        "background_tasks": "not_verified"}
            if attempt + 1 < self.finalize_attempts:
                time.sleep(self.finalize_interval)
        raise MemoryExecutionError("finalize", "completion_timeout", {
            "scope": "message_embeddings", "expected_messages": len(expected),
            "verified_messages": len(verified), "api_profile": self.api_profile})

    def reset(self) -> None:
        with execution_stage("reset"):
            if self._mode == "cloud":
                self.client.memory.delete(self._session_id)
                self._session_id = f"sess_{uuid.uuid4().hex[:8]}"
            else:
                self._ce_request("delete", f"sessions/{self._session_id}/memory", "reset", parse=False)
                self._session_id = f"sess_{uuid.uuid4().hex[:8]}"
                self._ce_request("post", "sessions", "reset", payload={
                    "session_id": self._session_id, "user_id": self._user_id}, parse=False)
            self._last_context = ""
            self._has_ingested = False
            self._expected_messages = {}
            self._ingest_batch = uuid.uuid4().hex
