# Optional evaluation services

<p>
  <a href="README.md"><img src="../assets/readme/language/zh.svg" alt="切换到中文" width="96" height="34"></a>
  <a href="README.en.md"><img src="../assets/readme/language/en-active.svg" alt="English (current language)" width="96" height="34"></a>
</p>

This directory provides local gateways, remote GPU services, and multi-system evaluation scripts. See [Quick start](../README.en.md#quick-start) for generation and evaluation with the four CLI configurations.

| File | Purpose | Configuration |
|---|---|---|
| [start_ingest_llm.sh](start_ingest_llm.sh) | Start remote vLLM over SSH and create a local tunnel | SSH host, user, credentials, and model path and ports in the script; requires Bash and sshpass |
| [run_eval.sh](run_eval.sh) | Start the combined gateway and run system evaluations | Set `BENCH_RUN`, `VLLM_UPSTREAM`, `EMBED_PORT`, and the storage services used by your systems |
| [embed_server.py](embed_server.py) | Serve local embeddings and forward LLM requests | Embedding model, upstream endpoint, and credentials; requires FastAPI, uvicorn, and sentence-transformers |
| [no_think_proxy.py](no_think_proxy.py) | Forward chat requests, process thinking output, and retry failures | Upstream endpoint and credentials; Python dependency: httpx |

Shell scripts use Bash and `./venv/bin/python`. Adjust host, model path, and port settings for your environment.

Dependencies: [core](../requirements-minimal.txt), [full evaluation](../requirements.txt), and [Mem0 integration](../requirements-memory-mem0.txt). The Mem0 list supplements the core environment.
