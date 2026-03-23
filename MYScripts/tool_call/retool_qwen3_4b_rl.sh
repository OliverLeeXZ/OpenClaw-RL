#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/mnt/shared-storage-user/llmbr-share/lixiaozhe/CODE/OpenClaw-RL"
SCRIPT_DIR="${PROJECT_ROOT}/MYScripts/tool_call"
SLIME_DIR="${PROJECT_ROOT}/slime"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${PROJECT_ROOT}/Megatron-LM}"
TOOLCALL_DIR="${PROJECT_ROOT}/toolcall-rl"

export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-1}"


VISIBLE_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

if (( VISIBLE_GPUS < NUM_GPUS )); then
    echo "Detected only ${VISIBLE_GPUS} visible GPUs, but NUM_GPUS=${NUM_GPUS} was requested."
    exit 1
fi

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_health_check_failure_threshold="${RAY_health_check_failure_threshold:-20}"
export RAY_health_check_period_ms="${RAY_health_check_period_ms:-5000}"
export RAY_health_check_timeout_ms="${RAY_health_check_timeout_ms:-30000}"
export RAY_num_heartbeats_timeout="${RAY_num_heartbeats_timeout:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"

DEBUG_KEEP_PROCS="${DEBUG_KEEP_PROCS:-0}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-4208}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
# This script is the TE path. Use the dedicated *_local.sh variant for local-only experiments.
TRAIN_TRANSFORMER_IMPL="transformer_engine"
TRAIN_ATTENTION_BACKEND="flash"
NVTE_DEBUG="${NVTE_DEBUG:-0}"
NVTE_DEBUG_LEVEL="${NVTE_DEBUG_LEVEL:-0}"
RAY_API_ADDRESS="http://127.0.0.1:8265"
RAY_JOB_SUBMISSION_ID="${RAY_JOB_SUBMISSION_ID:-retool-qwen3-4b-rl-$(date -u +%Y%m%d-%H%M%S)}"
JOB_SUBMITTED=0

cleanup() {
    set +e
    echo "Cleaning up Ray/SGLang processes..."
    ray stop --force >/dev/null 2>&1 || true
    pkill -f sglang >/dev/null 2>&1 || true
    pkill -f "ray::" >/dev/null 2>&1 || true
    pkill -f "train_async.py" >/dev/null 2>&1 || true
    pkill -f "torch/_inductor/compile_worker" >/dev/null 2>&1 || true
}

cleanup_on_exit() {
    exit_code=$?
    if [[ "${DEBUG_KEEP_PROCS}" == "1" ]]; then
        echo "DEBUG_KEEP_PROCS=1, skipping cleanup to preserve Ray/SGLang processes for inspection."
        return "${exit_code}"
    fi
    if [[ "${JOB_SUBMITTED}" == "1" && "${exit_code}" -ne 0 ]]; then
        echo "Ray job ${RAY_JOB_SUBMISSION_ID} exited with status ${exit_code}. Fetching logs before cleanup..."
        ray job logs --address="${RAY_API_ADDRESS}" "${RAY_JOB_SUBMISSION_ID}" 2>/dev/null || true
        ray job stop --address="${RAY_API_ADDRESS}" --no-wait "${RAY_JOB_SUBMISSION_ID}" >/dev/null 2>&1 || true
    fi
    cleanup
    return "${exit_code}"
}

trap cleanup_on_exit EXIT

cleanup

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

HF_CKPT="${HF_CKPT:-/mnt/shared-storage-user/llmbr-share/lixiaozhe/CKPT/qwen3-4b-sft}"
REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/CKPT/qwen3-4b-sft_torch_dist}"
SAVE_CKPT="${SAVE_CKPT:-${PROJECT_ROOT}/res_ckpt/tool_call/qwen3-4b-retool-rl/}"
RESUME_LOAD="${RESUME_LOAD:-${SAVE_CKPT}}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --load "${RESUME_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 20
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/shared-storage-user/llmbr-share/lixiaozhe/CODE/DATA/BytedTsinghua-SIA/DAPO-Math-17k/dapo_math_17k_cleaned.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-max-context-len 16384
   --rollout-temperature 1
   --num-steps-per-rollout 2
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /mnt/shared-storage-user/llmbr-share/lixiaozhe/CODE/DATA/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-max-context-len 32768
   --eval-top-p 1
   --eval-reward-key acc
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type k3
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

if [[ -n "${WANDB_KEY:-}" ]]; then
    WANDB_ARGS=(
       --use-wandb
       --wandb-project slime_retool
       --wandb-group qwen3-4B-rl_retool
       --wandb-key "${WANDB_KEY}"
    )
else
    WANDB_ARGS=()
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static 0.6
   --sglang-router-port "${SGLANG_ROUTER_PORT}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --no-persist-layer-norm
   --transformer-impl "${TRAIN_TRANSFORMER_IMPL}"
   --attention-backend "${TRAIN_ATTENTION_BACKEND}"
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

RUNTIME_ENV_JSON="{
  \"working_dir\": \"${SLIME_DIR}\",
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${TOOLCALL_DIR}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"NVTE_DEBUG\": \"${NVTE_DEBUG}\",
    \"NVTE_DEBUG_LEVEL\": \"${NVTE_DEBUG_LEVEL}\"
  }
}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SLIME_DIR=${SLIME_DIR}"
echo "MEGATRON_LM_PATH=${MEGATRON_LM_PATH}"
echo "TOOLCALL_DIR=${TOOLCALL_DIR}"
echo "NUM_GPUS=${NUM_GPUS}, ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}"
echo "HAS_NVLINK=${HAS_NVLINK}, SGLANG_ROUTER_PORT=${SGLANG_ROUTER_PORT}"
echo "TRAIN_TRANSFORMER_IMPL=${TRAIN_TRANSFORMER_IMPL}, TRAIN_ATTENTION_BACKEND=${TRAIN_ATTENTION_BACKEND}"
echo "RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=127.0.0.1 \
    --dashboard-port=8265

cd "${SLIME_DIR}"

JOB_SUBMITTED=1
echo "Submitting Ray job ${RAY_JOB_SUBMISSION_ID} to ${RAY_API_ADDRESS} ..."
ray job submit --address="${RAY_API_ADDRESS}" \
   --submission-id "${RAY_JOB_SUBMISSION_ID}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
