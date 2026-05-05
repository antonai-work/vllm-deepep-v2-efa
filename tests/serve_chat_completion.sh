#!/bin/bash
# vllm-deepep-v2-efa: end-to-end serving driver.
#
# Boots `vllm serve Qwen/Qwen3-30B-A3B-FP8` across 2 pods with the
# `--all2all-backend deepep_v2` native backend from vLLM PR #41183. One
# script, two roles (leader/worker) controlled by the ROLE env var.
#
# Env contract:
#   ROLE                   leader | worker           REQUIRED
#   MASTER_IP              pod-0 primary IP          REQUIRED
#   MODEL                  default Qwen/Qwen3-30B-A3B-FP8
#   DP_SIZE                default 16                 (8 per pod across 2 pods)
#   DP_SIZE_LOCAL          default 8                  (one GPU per DP rank)
#   TP_SIZE                default 1
#   DP_LEADER_RANK_START   default 8                 (worker --data-parallel-start-rank)
#   DP_MASTER_PORT         default 29500
#   VLLM_HOST_PORT         default 8000
#   MAX_MODEL_LEN          default 2048
#   MAX_NUM_SEQS           default 16
#   MAX_NUM_BATCHED_TOKENS default 256
#   GPU_MEM_UTIL           default 0.70
#
# Deploy pattern (from the operator machine, after both pods are Ready):
#
#   POD0_IP=$(kubectl -n vllm-deepep-v2-efa get pod vllm-deepep-v2-0 \
#       -o jsonpath='{.status.podIP}')
#   kubectl -n vllm-deepep-v2-efa exec -it vllm-deepep-v2-0 -- \
#       env MASTER_IP=$POD0_IP ROLE=leader bash /opt/tests/serve_chat_completion.sh
#   kubectl -n vllm-deepep-v2-efa exec -it vllm-deepep-v2-1 -- \
#       env MASTER_IP=$POD0_IP ROLE=worker bash /opt/tests/serve_chat_completion.sh
#
# Then from the operator machine:
#   kubectl -n vllm-deepep-v2-efa exec vllm-deepep-v2-0 -- \
#     curl -sS -X POST http://localhost:8000/v1/chat/completions \
#     -H 'content-type: application/json' \
#     -d '{"model":"Qwen/Qwen3-30B-A3B-FP8",
#          "messages":[{"role":"user","content":"hello"}],
#          "max_tokens":24}'
#
# Expected response shape: JSON with
#   choices[0].message.content   non-empty string
#   choices[0].finish_reason     "length" (hit max_tokens) or "stop"
#   usage.completion_tokens      <= 24

set -x
set -euo pipefail

ROLE="${ROLE:?Set ROLE=leader or ROLE=worker}"
MASTER_IP="${MASTER_IP:?Set MASTER_IP to pod-0 primary IP}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DP_SIZE="${DP_SIZE:-16}"
DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-8}"
TP_SIZE="${TP_SIZE:-1}"
DP_LEADER_RANK_START="${DP_LEADER_RANK_START:-8}"
DP_MASTER_PORT="${DP_MASTER_PORT:-29500}"
VLLM_HOST_PORT="${VLLM_HOST_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.70}"

# Pick the primary interface (enp* is the NCCL socket plane on p5.48xlarge
# / p5en.48xlarge; pods are hostNetwork=true in the reference manifest).
PRIMARY_IFACE=""
for n in /sys/class/net/enp*; do
  b=$(basename "$n")
  if [ -f "$n/operstate" ] && [ "$(cat $n/operstate)" = "up" ]; then
    PRIMARY_IFACE="$b"
    break
  fi
done
if [ -z "${PRIMARY_IFACE}" ]; then
  echo "WARN: no enp* interface in 'up' state; NCCL may pick the wrong iface" >&2
else
  export NCCL_SOCKET_IFNAME="$PRIMARY_IFACE"
  export GLOO_SOCKET_IFNAME="$PRIMARY_IFACE"
  export TP_SOCKET_IFNAME="$PRIMARY_IFACE"
fi

HOST_IP=$(hostname -I | awk '{print $1}')
export VLLM_HOST_IP="$HOST_IP"
export VLLM_DP_MASTER_IP="$MASTER_IP"
export VLLM_DP_MASTER_PORT="$DP_MASTER_PORT"
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_LOGGING_LEVEL=INFO

# PR #41183 env defaults - already baked into the image, but re-exported
# here for visibility.
export VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE="${VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE:-1}"
export VLLM_DEEPEP_V2_PREFER_OVERLAP="${VLLM_DEEPEP_V2_PREFER_OVERLAP:-0}"
export VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION="${VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION:-0}"

# Shim invariant: this repo's image ships WITHOUT /opt/api-shim. Force 0.
export DEEP_EP_USE_V2_SHIM=0

# vLLM imports MoE FP8 kernels from FlashInfer that require NVSHMEM/mlx5
# (not available on AWS EFA). Disable that code path.
export VLLM_USE_FLASHINFER_MOE_FP8=0

COMMON_ARGS=(
  "$MODEL"
  --tensor-parallel-size "$TP_SIZE"
  --data-parallel-size "$DP_SIZE"
  --data-parallel-size-local "$DP_SIZE_LOCAL"
  --data-parallel-address "$MASTER_IP"
  --data-parallel-rpc-port "$DP_MASTER_PORT"
  --data-parallel-backend mp
  --enable-expert-parallel
  --all2all-backend deepep_v2
  --enforce-eager
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --trust-remote-code
  --dtype bfloat16
)

case "$ROLE" in
  leader)
    echo "[serve_chat_completion.sh] role=leader host=$HOST_IP master=$MASTER_IP model=$MODEL"
    exec vllm serve "${COMMON_ARGS[@]}" \
      --host 0.0.0.0 \
      --port "$VLLM_HOST_PORT"
    ;;
  worker)
    echo "[serve_chat_completion.sh] role=worker host=$HOST_IP master=$MASTER_IP start-rank=$DP_LEADER_RANK_START"
    exec vllm serve "${COMMON_ARGS[@]}" \
      --data-parallel-start-rank "$DP_LEADER_RANK_START" \
      --headless
    ;;
  *)
    echo "ERROR: ROLE must be leader or worker (got '$ROLE')" >&2
    exit 2
    ;;
esac
