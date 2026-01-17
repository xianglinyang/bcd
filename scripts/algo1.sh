#!/bin/bash

# ----- 多卡 -----
# export CUDA_VISIBLE_DEVICES=0,1

# accelerate launch --multi_gpu --num_processes 2 src/train_algo1.py \
#   --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
#   --block_select cyclic --lr 1e-4 --beta 0.9 \
#   --total_outer_steps 2000 --mixed_precision bf16 




# ----- 单卡 -----
export CUDA_VISIBLE_DEVICES=1

accelerate launch --num_processes 1 src/train_algo1.py \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --block_select cyclic --lr 1e-4 --beta 0.9 \
  --total_outer_steps 2000 --mixed_precision bf16 

