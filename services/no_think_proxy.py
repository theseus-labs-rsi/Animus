"""聊天转发网关，使用 Python 标准库与 httpx。

处理 POST /v1/chat/completions：向非 DMXAPI 上游传递
chat_template_kwargs.enable_thinking=False 的默认值，移除响应中的
<think> 内容，并对超时和传输异常进行重试。

httpx 已包含在 requirements-minimal.txt 中。需要本地 embedding 时，
可使用 services.embed_server；其服务依赖收录在 requirements.txt。

用法：

    export LLM_UPSTREAM="$DEEPSEEK_BASE_URL"
    export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
    ./venv/bin/python -m services.no_think_proxy --port 9800

    export INGEST_LLM_BASE_URL=http://127.0.0.1:9800/v1
    ./venv/bin/python -m agent_harnesses preflight --experiment <toml>

上游地址也可读取 INGEST_LLM_BASE_URL 或 OPENAI_BASE_URL。凭据从环境变量
读取，--api-key-env 可指定变量名，默认 OPENAI_API_KEY。
上游连接设置 trust_env=False，直接连接配置的端点。
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
