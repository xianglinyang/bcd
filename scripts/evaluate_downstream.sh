#!/bin/bash

#------------- Hyperparameters to modify ---------------------
# 修改这部分的参数
gpu_num=1  # number of GPUs
available_gpu_ids=(1)
model_name_or_path_list=(
    "meta-llama/Llama-3.2-1B-Instruct"
)
# 把需要评估的dataset名称放在这里
dataset_name_list=(
    "mmlu"
    "gsm8k"
    "arc-c"
    "arc-e"
    "boolq"
    "MMLU-STEM"
    "sciq"
    "SimpleQA"
    "adv_glue"
    "aqua"
    "strategyqa"
)

# 一个dataset的eval_num
eval_num_per_dataset=100


# 用于清洗回答的llm模型名称，调用闭源的model。需要openai key或者openrouter api key。
# 需要提前配置好环境变量。

# Example:
# openai的model例子 gpt-4o-mini
# openrouter的model例子 openai/gpt-4o-mini
eval_llm_model_name="openai/gpt-4.1-nano"
# --- End of Hyperparameters to modify ------




# 下面的参数不要动
# --- preprocess-related hyperparameters ---
per_gpu_jobs_num=1
jobs_num=$((per_gpu_jobs_num*gpu_num))  # number of jobs to run in parallel, jobs_num = gpu_num*per_gpu_jobs_num

header="python -m src.downstream_evaluate.evaluate_common_reasoning"
base_arguments="\
--torch_type bf16 \
--split test \
--eval_num $eval_num_per_dataset \
--device cuda"

# -------------------------- Start of the Evaluation --------------------------
counter=0

for model_name_or_path in ${model_name_or_path_list[@]}; do
    for dataset_name in ${dataset_name_list[@]}; do
        gpu_id=${available_gpu_ids[$((counter % gpu_num))]}
        run_id=$RANDOM

        CUDA_VISIBLE_DEVICES=$gpu_id $header $base_arguments \
            --model_name_or_path $model_name_or_path \
            --dataset_name $dataset_name \
            --llm_model_name $eval_llm_model_name \
            --run_id $run_id &

        counter=$((counter + 1))

        if [ $((counter % jobs_num)) -eq 0 ]; then
            wait
        fi
    done
done

