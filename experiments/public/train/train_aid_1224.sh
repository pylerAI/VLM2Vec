#!/bin/bash
# NOTE: replace ... with actual paths
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# WANDB settings
export WANDB_DISABLED=false
export WANDB_RUN_GROUP="AID_GLOBAL_1224"
export WANDB_PROJECT="AiD-Global VLM2Vec"

# Experiment configurations: (CONFIG_NAME, USE_SUPCON_LOSS)
experiments=(
  "1224_general_template:True"
)

for exp_config in "${experiments[@]}"
do
  IFS=':' read -r CONFIG_NAME USE_SUPCON_LOSS <<< "$exp_config"

  export EXP_NAME=aid_global_${CONFIG_NAME}
  if [ "$USE_SUPCON_LOSS" = "False" ]; then
    export EXP_NAME=${EXP_NAME}_no_supcon
  fi
  export WANDB_NAME=$EXP_NAME
  export EXP_DIR=/gpfs/private/iji/experiments/pyler_embeds/VLM2Vec/$WANDB_RUN_GROUP/$EXP_NAME
  export WANDB_DIR=$EXP_DIR

  echo "=========================================="
  echo "Running experiment: $EXP_NAME"
  echo "Config: $CONFIG_NAME"
  echo "Use SupCon Loss: $USE_SUPCON_LOSS"
  echo "Experiment directory: $EXP_DIR"
  echo "=========================================="

  mkdir -p $EXP_DIR/wandb
  rm -rf $EXP_DIR/wandb/*

  # Build use_supcon_loss flag
  SUPCON_FLAG=""
  if [ "$USE_SUPCON_LOSS" = "True" ]; then
    SUPCON_FLAG="--use_supcon_loss True"
  fi

  cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 --master_port=2207 --max_restarts=0 train.py \
    --lora \
    --lora_r 16 \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --checkpoint_path /gpfs/public/artifacts/models/VLM2Vec/VLM2Vec-V2.0 \
    --bf16 \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --dataloader_num_workers 8 \
    --dataset_config experiments/public/train/${CONFIG_NAME}.yaml \
    --image_dir /gpfs/public/artifacts/aid-global/v0.1.0/train/all/image \
    --run_name $EXP_NAME \
    --output_dir $EXP_DIR \
    --grad_cache True \
    --per_device_train_batch_size 256 \
    --gc_q_chunk_size 8 \
    --gc_p_chunk_size 8 \
    --interleave_batch_size 64 \
    --lr_scheduler_type linear \
    --learning_rate 5e-5 \
    --max_steps 1000 \
    --warmup_steps 50 \
    --save_steps 200 \
    --logging_steps 1 \
    --save_safetensors True \
    --remove_unused_columns False \
    --resume_from auto \
    --report_to wandb \
    $SUPCON_FLAG \
    2>&1 | tee $EXP_DIR/train.log"

    echo $cmd
    eval $cmd
done
