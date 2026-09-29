#!/usr/bin/env bash
# 多系统记忆评测 · 单一入口(103.1 冷启动可跑)。
#
# 三链分离:
#   ingest(mem0 抽取 / amem 笔记演化 / zep 实体抽取) → 本地 vLLM Qwen3-8B,
#       但【统一经 embed_server 网关 :9800】,不直连 :8000。
#   答题 + 判分(unified_answer / judge)             → DMXAPI DeepSeek-V4-Flash(国内直连)。
#   embedding(baseline / mem0 / zep / amem)          → 本地 bge-small-zh-v1.5(embed_server)。
#
# 为什么所有 ingest LLM 走网关:Qwen3 默认开 thinking → 机械抽取慢 ~9× 且 <think> 撑爆
# max_tokens 截断成无效 JSON。no-think / 剥 <think> / 超时重试 都在【网关一处】做,
# adapter 不再各自打补丁(根因修复,非打地鼠)。新接系统只要指向 :9800 即自动获得全部加固。
#
# 用法:
#   bash services/run_eval.sh                                  # 默认 6 系统,workers=2
#   bash services/run_eval.sh --systems A,B,C,mem0 --workers 2 # 自定义(透传给 multi_system)
#   BENCH_RUN=output/runs/<run> bash services/run_eval.sh      # 换评测集
set -uo pipefail
cd "$(dirname "$0")/.."                       # → 项目根

VENV=./venv/bin/python
EMBED_PORT="${EMBED_PORT:-9800}"
VLLM_UPSTREAM="${VLLM_UPSTREAM:-http://localhost:8000/v1}"   # 本地 vLLM(ingest 真上游)

# ── 1. 全本地 + 国内直连,绝不走代理(Mihomo 拦局域网 = 502 的根源) ──
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="*" no_proxy="*"

# ── 2. 关第三方遥测(mem0/chromadb 打 us.i.posthog.com 不通会拖死启动) ──
export MEM0_TELEMETRY=False ANONYMIZED_TELEMETRY=False DO_NOT_TRACK=1 POSTHOG_DISABLED=1

# ── 3. ingest 链统一指向网关(:9800),不是直连 vLLM ──
export INGEST_LLM_BASE_URL="http://localhost:${EMBED_PORT}/v1"
export INGEST_LLM_MODEL="${INGEST_LLM_MODEL:-Qwen3-8B}"
export QDRANT_PORT="${QDRANT_PORT:-6343}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"  # bge 已缓存,免 HF 限流警告

gw_ok() { curl -s --noproxy '*' --max-time 3 "http://localhost:${EMBED_PORT}/v1/models" >/dev/null 2>&1; }

# ── 4. 确保 embed_server 网关在跑;上游用 LLM_UPSTREAM 钉死 vLLM(优先级最高,防自指) ──
if ! gw_ok; then
  echo "[run_eval] 启动 embed_server 网关 :${EMBED_PORT}(上游=${VLLM_UPSTREAM})…"
  LLM_UPSTREAM="${VLLM_UPSTREAM}" nohup $VENV -m services.embed_server --port "${EMBED_PORT}" \
    > embed_server.log 2>&1 &
  for _ in $(seq 1 20); do gw_ok && break; sleep 2; done
fi
gw_ok && echo "[run_eval] 网关 OK :${EMBED_PORT} → ${VLLM_UPSTREAM}" \
      || { echo "[run_eval] ✗ embed_server 网关起不来,看 embed_server.log"; exit 1; }

# ── 5. 跑评测(默认 6 系统;命令行参数透传覆盖) ──
R="${BENCH_RUN:-output/runs/legal__20260616-112358}"
ARGS=("$@"); [ ${#ARGS[@]} -eq 0 ] && ARGS=(--systems A,B,C,mem0,zep,amem --workers 2)
echo "[run_eval] 评测集 $R | 参数 ${ARGS[*]}"
exec $VENV -u -m eval.multi_system \
  --bench "$R/06_grounded_questions.json" --corpus "$R/05_corpus.json" "${ARGS[@]}"
