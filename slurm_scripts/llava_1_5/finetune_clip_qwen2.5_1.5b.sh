#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --mail-user=xiangyan@comp.nus.edu.sg
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus=h100-96:2
#SBATCH --partition=gpu-long
#SBATCH --time=02-00:00:00
#SBATCH --output=logs/llava1.5-fintune.out


# Activate conda environment
source ~/miniconda3/bin/activate llava-next # source ~/miniconda3/bin/activate llava

which python
python --version
nvidia-smi
cd /home/x/xiangyan/vlm_dissection/code/LLaVA-NeXT/

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0  # Comment this out to let NCCL auto-detect
export NCCL_DEBUG=INFO
export WANDB_API_KEY="454cd689208c89f63deeb877aed3fba714fce4c6"

LLM_VERSION="Qwen/Qwen2.5-1.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION="qwen_1_5"


BASE_RUN_NAME="llava1_5-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
SFT_RUN_NAME="llava1_5-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-finetune_llava_next_data_qwen1_5"
echo "SFT_RUN_NAME: ${SFT_RUN_NAME}"

srun --ntasks=1 --cpus-per-task=16 --gpus=2 \
    torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29500 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ~/vlm_dissection/data/llava_next_sft_data.json \
    --image_folder ~/vlm_dissection/data/llava_next_sft_images \
    --pretrain_mm_mlp_adapter ~/vlm_dissection/code/LLaVA-NeXT/checkpoints/llava1.5-qwen2.5-1.5b-clip-pretrain/mm_projector.bin \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --run_name $SFT_RUN_NAME \
    --output_dir ./checkpoints/${SFT_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa