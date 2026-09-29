#!/bin/bash
# 在 GPU 机器 (192.168.103.1) 上启动 vLLM serving Qwen3-8B，
# 然后在本机 (Mac) 上开 SSH 隧道，让 INGEST_LLM_BASE_URL 指向它。
#
# 用法：
#   bash services/start_ingest_llm.sh        # 启动远端 vLLM + 本机隧道
#   bash services/start_ingest_llm.sh stop   # 停止
#
# 启动后设置环境变量：
#   export INGEST_LLM_BASE_URL=http://localhost:18000/v1
#   export INGEST_LLM_MODEL=Qwen3-8B

REMOTE_USER=${INGEST_SSH_USER:-ziqian}
REMOTE_HOST=${INGEST_SSH_HOST:-192.168.103.1}
REMOTE_PASS=${INGEST_SSH_PASS:?'请设置 INGEST_SSH_PASS 环境变量(或用 SSH key 免密)'}
REMOTE_PORT=8000
LOCAL_PORT=18000
MODEL_PATH=/data/models/Qwen3-8B
GPU_ID=0

if [ "$1" = "stop" ]; then
    echo "停止 SSH 隧道..."
    pkill -f "ssh.*-L ${LOCAL_PORT}:localhost:${REMOTE_PORT}.*${REMOTE_HOST}" 2>/dev/null
    echo "停止远端 vLLM..."
    sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} \
        'pkill -f "vllm.entrypoints.openai.api_server.*Qwen3-8B"' 2>/dev/null
    echo "done"
    exit 0
fi

echo "=== 1/3 在远端启动 vLLM (GPU ${GPU_ID}, ${MODEL_PATH}) ==="
sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} \
    "source ~/vllm_env/bin/activate && \
     CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -m vllm.entrypoints.openai.api_server \
       --model ${MODEL_PATH} \
       --served-model-name Qwen3-8B \
       --port ${REMOTE_PORT} \
       --trust-remote-code \
       --max-model-len 32768 \
       --gpu-memory-utilization 0.90 \
       > ~/vllm_serve.log 2>&1 &"
echo "远端 vLLM 已后台启动，日志: ~/vllm_serve.log"

echo "=== 2/3 等待远端 vLLM 就绪 ==="
for i in $(seq 1 60); do
    if sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} \
        "curl -s http://localhost:${REMOTE_PORT}/v1/models 2>/dev/null" | grep -q model; then
        echo "vLLM 就绪 (${i}s)"
        break
    fi
    sleep 3
done

echo "=== 3/3 建立 SSH 隧道 (本机 :${LOCAL_PORT} → 远端 :${REMOTE_PORT}) ==="
sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no -N \
    -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} &
sleep 2

if curl -s http://localhost:${LOCAL_PORT}/v1/models | grep -q model; then
    echo ""
    echo "✓ 隧道就绪！设置环境变量："
    echo "  export INGEST_LLM_BASE_URL=http://localhost:${LOCAL_PORT}/v1"
    echo "  export INGEST_LLM_MODEL=Qwen3-8B"
else
    echo "⚠ 隧道未通，检查远端日志: ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail ~/vllm_serve.log'"
fi
