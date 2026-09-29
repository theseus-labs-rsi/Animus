# 可选评测服务

<p>
  <a href="README.md"><img src="../assets/readme/language/zh-active.svg" alt="中文（当前语言）" width="96" height="34"></a>
  <a href="README.en.md"><img src="../assets/readme/language/en.svg" alt="Switch to English" width="96" height="34"></a>
</p>

本目录提供本地网关、远程 GPU 服务和多系统评测入口。生成与四模型 CLI 试答见[快速开始](../README.md#快速开始)。

| 文件 | 用途 | 运行前配置 |
|---|---|---|
| [start_ingest_llm.sh](start_ingest_llm.sh) | 通过 SSH 启动远端 vLLM，并建立本地隧道 | SSH 主机、用户、凭据，以及脚本内的模型路径和端口；依赖 Bash 与 sshpass |
| [run_eval.sh](run_eval.sh) | 启动联合网关并执行多系统评测 | 用 `BENCH_RUN` 指定评测集，配置 `VLLM_UPSTREAM`、`EMBED_PORT` 及所用系统的存储服务 |
| [embed_server.py](embed_server.py) | 本地 embedding 与 LLM 转发的联合网关 | embedding 模型、上游 LLM 地址与凭据；依赖 FastAPI、uvicorn、sentence-transformers |
| [no_think_proxy.py](no_think_proxy.py) | 聊天转发、thinking 处理与重试 | 上游 LLM 地址与凭据；Python 依赖为 httpx |

Shell 脚本使用 Bash 和 `./venv/bin/python`。运行前按实际环境调整脚本中的主机、模型路径与端口。

## 依赖

- [requirements-minimal.txt](../requirements-minimal.txt)：生成和原生 CLI 控制面的 Python 依赖，包含 httpx。
- [requirements.txt](../requirements.txt)：完整评测环境，包含联合网关与多个系统适配器的依赖。
- [requirements-memory-mem0.txt](../requirements-memory-mem0.txt)：Mem0 接入的定版本依赖，配合基础环境使用。
