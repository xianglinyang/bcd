#!/bin/bash

# ----- 多卡 -----
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --config_file src/fsdp_config/llama_fsdp_config.yaml src/train_algo1.py \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --dataset_name alpaca \
  --train_batch_size 1 \
  --num_workers 2 \
  --max_length 2048 \
  --block_select cyclic \
  --lr 1e-4 \
  --beta 0.9 \
  --total_outer_steps 2000 \
  --log_every 20 \
  --seed 42 \
  --mixed_precision bf16 \
  --save_dir runs/algo1


# ----- 单卡 -----
# export CUDA_VISIBLE_DEVICES=1

# accelerate launch --config_file src/fsdp_config/llama_fsdp_config.yaml src/train_algo1.py \
#   --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
#   --dataset_name alpaca \
#   --train_batch_size 1 \
#   --num_workers 2 \
#   --max_length 2048 \
#   --block_select cyclic \
#   --lr 1e-4 \
#   --beta 0.9 \
#   --total_outer_steps 2000 \
#   --log_every 20 \
#   --seed 42 \
#   --mixed_precision bf16 \
#   --save_dir runs/algo1

