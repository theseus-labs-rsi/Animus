"""no-think chat 网关（stdlib + httpx）—— `services/embed_server.py` 的无依赖替代品。

用途与 `embed_server.py` 的 `POST /v1/chat/completions` 完全一致：

- 对非 DMXAPI 上游注入 `chat_template_kwargs.enable_thinking=False`；
- 剥掉响应里的 `<think>…</think>`（兜底：上游不认该字段时仍有保护）；
- 超时重试 3 次。

**为什么需要它**：`embed_server.py` 依赖 `fastapi`/`uvicorn`，而这两个包不在任何
requirements 里（只在 docker-compose 的容器里可用），macOS venv 起不来。当被测系统
的 embedding 不走网关时（例如 mem0 用 `huggingface` provider 在本地跑 bge），
就只需要 chat 转发这一半功能，本文件即可满足，且不引入新依赖。

**为什么必须关 thinking**：推理模型的 `reasoning_content` 会吃掉补全预算，
机械抽取（带 `response_format=json_object`）的结果被截断成非法 JSON，触发 adapter 的
json 守卫 fail-closed，整个 ingest 中止。长语料下必然发生，且调大 `max_tokens` 只能把
失败点往后推（实测 2000 → 第 27 篇，8000 → 第 114 篇）。关掉 thinking 让
`reasoning_tokens` 归 0，是唯一确定性修法。详见
`output/local_services/mem0_smoke_report.md` 的三次 run 对照。

用法：

    # 上游地址与 key 只走环境变量（见 services/run_eval.sh 的约定）
    export LLM_UPSTREAM="$DEEPSEEK_BASE_URL"     # 或 INGEST_LLM_BASE_URL / OPENAI_BASE_URL
    export OPENAI_API_KEY="$DEEPSEEK_API_KEY"    # 或 --api-key-env 指定别的变量名
    ./venv/bin/python -m services.no_think_proxy --port 9800

    # 让 harness 的 ingest LLM 指向它（secrets.env 或 shell export 均可）
    export INGEST_LLM_BASE_URL=http://127.0.0.1:9800/v1
    ./venv/bin/python -m agent_harnesses preflight --experiment <toml>

安全：**不要把 API key 写在命令行参数上** —— `ps`/`pgrep -fl` 会把 argv 暴露给同机
其它进程。key 只从环境变量读（`--api-key-env`，默认 `OPENAI_API_KEY`）。

网络：上游调用固定 `trust_env=False`，**不走 `HTTP(S)_PROXY`**。与
`services/run_eval.sh`「unset 全部代理、绝不走代理」的约定一致 —— 经代理会间歇性
`httpx.ProxyError: 502`，直连实测还快约 6×。传输层异常（Timeout / Connect / Read /
Proxy）统一重试 3 次并退避，不会穿出循环崩掉 handler。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import httpx

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)

UPSTREAM = ""
API_KEY = ""
STATS = {"requests": 0, "injected": 0, "errors": 0}


def _path_of(raw_path: str) -> str:
    """同时接受 origin-form (`/v1/...`) 与 absolute-form (`http://host/v1/...`)。

    OpenAI SDK 经 HTTP 代理访问 localhost 时会发 absolute-form URI，
    此时 `self.path` 是完整 URL，直接 `startswith("/v1/...")` 会全部落空 → 404。
    """
    return urlsplit(raw_path).path or "/"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 - 静音默认 access log，只用 _log
        pass

    def _log(self, message: str) -> None:
        print(f"[no-think-proxy] {message}", flush=True)

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler 接口
        path = _path_of(self.path)
        if path.startswith("/v1/models"):
            # preflight 的网关可达性探测打的就是这个端点。
            return self._send(200, {"object": "list",
                                    "data": [{"id": "no-think-proxy", "object": "model"}]})
        if path.startswith("/stats"):
            return self._send(200, STATS)
        return self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler 接口
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw or b"{}")
        except Exception:
            return self._send(400, {"error": {"message": "bad json"}})

        if not _path_of(self.path).startswith("/v1/chat/completions"):
            self._log(f"UNHANDLED PATH {self.path} -> 404 (keys={sorted(data.keys())})")
            return self._send(404, {"error": {"message": "not found"}})

        STATS["requests"] += 1
        ctk = dict(data.get("chat_template_kwargs") or {})
        ctk.setdefault("enable_thinking", False)
        data["chat_template_kwargs"] = ctk
        STATS["injected"] += 1

        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"

        last_err = None
        for attempt in range(3):
            try:
                # trust_env=False：上游调用不走 HTTP(S)_PROXY。
                # 仓库约定就是 ingest 链直连上游（services/run_eval.sh 明确
                # unset http_proxy/https_proxy，注释写着「绝不走代理」）；
                # 而且经沙箱代理会间歇性抛 httpx.ProxyError: 502，直连实测还快约 6×。
                response = httpx.post(
                    f"{UPSTREAM}/chat/completions",
                    json=data,
                    headers=headers,
                    timeout=120,
                    trust_env=False,
                )
                body = response.json()
                self._log(
                    f"req#{STATS['requests']} keys={sorted(data.keys())} "
                    f"tools={'tools' in data} -> upstream {response.status_code}"
                )
                if response.status_code != 200:
                    # 只截断记录上游错误体，便于排障；不落盘任何响应正文。
                    self._log(f"  upstream error body: {json.dumps(body)[:300]}")
                if response.status_code == 200:
                    for choice in body.get("choices", []):
                        message = choice.get("message") or choice.get("delta") or {}
                        content = message.get("content")
                        if content and "<think>" in content:
                            message["content"] = _THINK_OPEN_RE.sub(
                                "", _THINK_RE.sub("", content)
                            )
                return self._send(response.status_code, body)
            except httpx.TransportError as exc:
                # 捕获整个传输层异常族（Timeout / Connect / Read / Proxy / …）。
                # 只 catch TimeoutException + ConnectError 的话，ProxyError 会穿出循环
                # 直接崩掉 handler —— 客户端拿到的是断连，而不是可重试的错误，
                # 上游侧表现为 mem0 的 operation_failed（看起来像模型/解析问题，实为传输层）。
                last_err = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        STATS["errors"] += 1
        self._log(f"req#{STATS['requests']} upstream timeout: {last_err}")
        return self._send(504, {"error": {"message": f"upstream timeout: {last_err}"}})


def _resolve_upstream(explicit: str | None) -> str:
    return (
        explicit
        or os.getenv("LLM_UPSTREAM")
        or os.getenv("INGEST_LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    ).rstrip("/")


def main(argv: list[str] | None = None) -> int:
    global UPSTREAM, API_KEY  # noqa: PLW0603 - 进程级单例配置
    parser = argparse.ArgumentParser(description="no-think chat 网关（stdlib，无 fastapi 依赖）")
    parser.add_argument("--port", type=int, default=9800)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--upstream",
        default=None,
        help="默认取 $LLM_UPSTREAM / $INGEST_LLM_BASE_URL / $OPENAI_BASE_URL",
    )
    # 不要把 key 写在命令行上：ps 会把 argv 暴露给同机其它进程。
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="持有上游 key 的环境变量名",
    )
    args = parser.parse_args(argv)

    UPSTREAM = _resolve_upstream(args.upstream)
    API_KEY = os.getenv(args.api_key_env, "")
    if not UPSTREAM:
        print("error: 缺上游地址：传 --upstream 或设 $LLM_UPSTREAM")
        return 2
    if not API_KEY:
        print(f"error: 缺上游 key：设 ${args.api_key_env}")
        return 2

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[no-think-proxy] listening on {args.host}:{args.port} -> {UPSTREAM}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
