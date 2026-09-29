"""
OpenAI-compatible 本地 embedding + LLM 代理。

  POST /v1/embeddings       → 本地 BAAI/bge-small-zh-v1.5 (512维)
  POST /v1/chat/completions  → 透传上游 LLM (DMXAPI / INGEST_LLM)
  GET  /v1/models            → 模型列表

Zep CE 的 embedding 和 LLM 共用一个 openai_endpoint，无法分别配端点。
本服务统一接收 zep 的所有 OpenAI 调用：embedding 本地处理、LLM 转发上游。

启动:  ./venv/bin/python -m services.embed_server [--port 9800]
"""
from __future__ import annotations
import argparse, asyncio, os, re
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    from sentence_transformers import SentenceTransformer
    # 钉死 CPU:bge-small 很轻,放 CPU 反而稳——否则它默认抢 GPU0,与 ingest vLLM 争算力,
    # zep ingest 期 embedding 被 vLLM 饱和的 GPU0 卡到超时(context deadline exceeded)。
    _model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")
    dim = (_model.get_embedding_dimension()
           if hasattr(_model, "get_embedding_dimension")
           else _model.get_sentence_embedding_dimension())
    print(f"[embed_server] bge-small-zh-v1.5 loaded, dim={dim}")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    data = await request.json()
    texts = data.get("input", [])
    if isinstance(texts, str):
        texts = [texts]
    # encode 是阻塞 CPU 调用,丢线程池跑,别卡住事件循环(否则并发时 chat 代理 + embedding 互相饿死)
    vecs = await asyncio.to_thread(
        _model.encode, texts, normalize_embeddings=True, show_progress_bar=False)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": v.tolist(), "index": i}
            for i, v in enumerate(vecs)
        ],
        "model": "bge-small-zh-v1.5",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    upstream = os.getenv(
        "LLM_UPSTREAM",
        os.getenv("INGEST_LLM_BASE_URL",
                   os.getenv("OPENAI_BASE_URL", "https://www.dmxapi.cn/v1")),
    )
    upstream = upstream.rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "")
    data = await request.json()
    # 上游是本地 vLLM(Qwen3)时关 thinking:默认 thinking on 会让机械抽取慢 ~9× 且
    # <think> 撑爆 max_tokens 截断成空内容。注入 chat_template_kwargs.enable_thinking=False
    # (vLLM 顶层接受);DMXAPI 不认该字段,故仅对非 DMXAPI 上游注入。下方 <think> 剥离保留兜底。
    if "dmxapi" not in upstream.lower():
        ctk = dict(data.get("chat_template_kwargs") or {})
        ctk.setdefault("enable_thinking", False)
        data["chat_template_kwargs"] = ctk
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_err = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{upstream}/chat/completions",
                    json=data,
                    headers=headers,
                )
                body = resp.json()
                if resp.status_code == 200:
                    for ch in body.get("choices", []):
                        msg = ch.get("message") or ch.get("delta") or {}
                        c = msg.get("content")
                        if c and "<think>" in c:
                            cleaned = _THINK_RE.sub("", c)
                            msg["content"] = _THINK_OPEN_RE.sub("", cleaned)
                return JSONResponse(content=body, status_code=resp.status_code)
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_err = e
            if attempt < 2:
                import asyncio
                await asyncio.sleep(2 ** attempt)
    return JSONResponse(
        content={"error": {"message": f"upstream timeout after 3 retries: {last_err}"}},
        status_code=504,
    )


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": "bge-small-zh-v1.5", "object": "model", "owned_by": "local"}],
    }


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9800)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
