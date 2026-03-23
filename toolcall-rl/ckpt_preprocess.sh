export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-1}"
cd /mnt/shared-storage-user/llmbr-share/lixiaozhe/CODE/OpenClaw-RL/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/mnt/shared-storage-user/llmbr-share/lixiaozhe/CODE/OpenClaw-RL/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /mnt/shared-storage-user/llmbr-share/lixiaozhe/CKPT/qwen3-4b-sft \
    --rotary-base 5000000 \
    --save /mnt/shared-storage-user/llmbr-share/lixiaozhe/CODE/OpenClaw-RL/CKPT/qwen3-4b-sft_torch_dist
