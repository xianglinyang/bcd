'''
Evaluate the loss of the fine-tuned model on different general capability datasets

Open LLM Leaderboard
MT benchmark
MMLU (Hendrycks et al., 2020), 
ARC (Clark et al., 2018), 
GSM8K (Cobbe et al., 2021), and
TRUTHFULQA (Bisk et al., 2020)
BBH (Bisk et al., 2020)
humaneval
alpaca
'''

import os
import json
import time
from datetime import datetime
import random
import logging
import torch
import argparse
import numpy as np
import fcntl
import asyncio
from typing import Dict
from src.downstream_evaluate.llm_evaluator import load_judge_model
from src.downstream_evaluate.llm_evaluator import JudgeType
from src.downstream_evaluate.llm_zoo.code_base_models import VLLMModel
from src.downstream_evaluate.reasoning_datasets import ReasoningDataset

logger = logging.getLogger(__name__)

def save_results(results: Dict, path="eval_results"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    save_file = os.path.join(path, "common_ability.json")
    max_retries = 5
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Open in r+ mode (read and write without truncating)
            with open(save_file, 'r+' if os.path.exists(save_file) else 'w+') as f:
                # Acquire lock before doing anything
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    try:
                        # Read existing content
                        f.seek(0)  # Ensure we're at the start of file
                        existing_evaluation = json.load(f)
                    except (ValueError, json.JSONDecodeError):
                        # Handle empty or invalid file
                        existing_evaluation = []
                    
                    # Append new results
                    existing_evaluation.append(results.copy())
                    
                    # Write back entire content
                    f.seek(0)  # Go back to start
                    f.truncate()  # Clear existing content
                    json.dump(existing_evaluation, f, indent=4)
                    
                    print(f"Evaluation results saved at {save_file}")
                    return True
                    
                finally:
                    # Release the lock
                    print("Releasing lock...")
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    print("Lock released")
                    
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to save results after {max_retries} attempts: {e}")
                return False
            time.sleep(retry_delay)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--eval_num", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--torch_type", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="MTBench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--llm_model_name", type=str, default="openai/gpt-4.1-nano")
    args = parser.parse_args()

    # log the args
    logger.info("Evaluate MT Bench")
    logger.info(f"Arguments: {args}")

    # read args
    model_name_or_path = args.model_name_or_path
    torch_type = args.torch_type
    eval_num = args.eval_num
    device = args.device
    tensor_parallel_size = args.tensor_parallel_size
    judge_model_name = args.llm_model_name
    if torch_type == "bf16":
        torch_type = torch.bfloat16
    elif torch_type == "fp16":
        torch_type = torch.float16
    elif torch_type == "fp32":
        torch_type = torch.float32
    else:
        raise ValueError(f"Invalid torch_type: {torch_type}")

    llm = VLLMModel(model_name_or_path=model_name_or_path, torch_dtype=torch_type, device=device, tensor_parallel_size=tensor_parallel_size)
    dataset = ReasoningDataset(dataset_name="mtbench", split="test")
    questions = [item[0] for item in dataset]
    # Determine actual evaluation number
    if eval_num == -1:
        actual_eval_num = len(dataset)
        eval_idxs = list(range(len(dataset)))
    elif eval_num > len(dataset):
        actual_eval_num = len(dataset)
        eval_idxs = list(range(len(dataset)))
    else:
        eval_idxs = random.sample(range(len(dataset)), eval_num)
        actual_eval_num = eval_num
    selected_questions = [questions[idx] for idx in eval_idxs]
    
    responses, latency_metrics = llm.batch_invoke(selected_questions, return_latency=True)
    
    mtbench_judge = load_judge_model(JudgeType.MT_BENCH, judge_model_name)
    scores, _ = mtbench_judge.batch_get_score(selected_questions, responses)

    results = {
        "avg_scores": np.mean(scores),
        "dataset_name": "MTBench",
        "model_name_or_path": model_name_or_path,
        "eval_num": actual_eval_num,
        "tensor_parallel_size": tensor_parallel_size,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latency_metrics": latency_metrics,
    }
    logger.info(f"Evaluation results: {results}")
    save_results(results)
    print("End of evaluation")


if __name__ == "__main__":
    asyncio.run(main())
