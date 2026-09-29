#!/usr/bin/env bash
# 过夜批量合成 6 个场景 benchmark:game / agent / cs / companion / assistant / kb。
# 顺序跑(共用 DMXAPI DeepSeek-V4-Flash 强模型,避免并发打爆尾延迟);单个失败不阻断其余。
# 生成只依赖 DMXAPI(国内直连),不碰 vLLM / qdrant / embed_server(那些是评测 ingest 用的)。
#
# 用法:  nohup bash services/run_overnight_gen.sh > output/eval/overnight_gen.out 2>&1 &
#        MINQ=80 MTOK=0.4 ROUNDS=2 bash services/run_overnight_gen.sh    # 调规模
set -u
cd "$(dirname "$0")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy   # DMXAPI 国内直连,勿代理
export NO_PROXY="*" no_proxy="*"
VENV=./venv/bin/python
# 默认 6 场景;给了参数则只跑指定的(如 `bash run_overnight_gen.sh game cs assistant` 重跑失败的)
SCENARIOS=("$@"); [ ${#SCENARIOS[@]} -eq 0 ] && SCENARIOS=(game agent cs companion assistant kb)
MINQ="${MINQ:-66}"; MTOK="${MTOK:-0.3}"; ROUNDS="${ROUNDS:-1}"
LOGDIR=output/eval; mkdir -p "$LOGDIR"
BATCH="$LOGDIR/overnight_gen_batch.log"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$BATCH"; }

say "=== 过夜批量合成启动 | 6 场景 | min-q=$MINQ target=${MTOK}M max-rounds=$ROUNDS ==="

# 预检:DMXAPI 可达(生成全靠它),不通就别白跑一夜
if ! $VENV -c "import config;config.client.chat.completions.create(model='DeepSeek-V4-Flash',messages=[{'role':'user','content':'好'}],max_tokens=3)" >/dev/null 2>&1; then
  say "✗ DMXAPI 预检失败,中止(检查 .env / 网络)"; exit 1
fi
say "✓ DMXAPI 预检通过,开始逐场景合成"

ok=0; fail=0
for sc in "${SCENARIOS[@]}"; do
  say ">>> [$sc] 开始"
  t0=$(date +%s)
  if $VENV -u -m pipeline.factory --scenario "$sc" --min-questions "$MINQ" --target-mtokens "$MTOK" \
        --max-rounds "$ROUNDS" --tag overnight >> "$LOGDIR/gen_${sc}.log" 2>&1; then
    say ">>> [$sc] 完成($(( $(date +%s) - t0 ))s)"; ok=$((ok+1))
  else
    say ">>> [$sc] 失败(继续下一个;详见 $LOGDIR/gen_${sc}.log)"; fail=$((fail+1))
  fi
done

say "=== 全部结束:成功 $ok / 失败 $fail。run 汇总(tag=overnight): ==="
$VENV -m pipeline.factory --list-runs 2>/dev/null | grep -E "run_id|overnight" | tee -a "$BATCH"
