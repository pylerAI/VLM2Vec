#!/bin/bash
# NOTE: replace ... with actual paths
# export LD_LIBRARY_PATH=...
# export PATH=...
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# export HF_DATASETS_CACHE=...
# export HF_HOME=...
export WANDB_DISABLED=false
# export WANDB_PROJECT=...
# export WANDB_API_KEY=...
# export HUGGING_FACE_HUB_TOKEN=...
# export WANDB_PROJECT=...
# export WANDB_RUN_GROUP=...
export EXP_NAME=aid_global_qwen2vl-2B_1103

export WANDB_NAME=$EXP_NAME
export EXP_DIR=/gpfs/private/iji/experiments/pyler_embeds/VLM2Vec/$EXP_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

# cd /gpfs/private/iji/YT-Foundation/my_files/VLM2Vec
# cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2207 --max_restarts=0 train.py 
# --lora 
# --lora_r 16 
# --model_name Qwen/Qwen2-VL-2B-Instruct 
# --bf16 
# --pooling eos 
# --normalize True 
# --temperature 0.02 
# --dataloader_num_workers 8 
# --dataset_config experiments/public/train/train_aid_global.yaml 
# --run_name $EXP_NAME 
# --output_dir $EXP_DIR 
# --grad_cache True 
# --per_device_train_batch_size 512 
# --gc_q_chunk_size 8 
# --gc_p_chunk_size 8 
# --interleave_batch_size 64 
# --lr_scheduler_type linear 
# --learning_rate 5e-5 
# --max_steps 5000 
# --warmup_steps 100 
# --save_steps 500 
# --logging_steps 1 
# --save_safetensors True 
# --remove_unused_columns False 
# --resume_from auto 
# --report_to wandb 2>&1 | tee $EXP_DIR/train.log"
cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2207 --max_restarts=0 train.py --lora --lora_r 16 --model_name Qwen/Qwen2-VL-2B-Instruct --checkpoint_path /gpfs/public/artifacts/models/VLM2Vec/VLM2Vec-V2.0 --bf16 --pooling eos --normalize True --temperature 0.02 --dataloader_num_workers 8 --dataset_config experiments/public/train/train_aid_global.yaml --image_dir /gpfs/public/artifacts/aid-global/v0.1.0/train/all/image --run_name $EXP_NAME --output_dir $EXP_DIR --grad_cache True --per_device_train_batch_size 128 --gc_q_chunk_size 8 --gc_p_chunk_size 8 --interleave_batch_size 64 --lr_scheduler_type linear --learning_rate 5e-5 --max_steps 1000 --warmup_steps 50 --save_steps 200 --logging_steps 1 --save_safetensors True --remove_unused_columns False --resume_from auto --report_to tensorboard 2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd
