'''
Algo2 / L-MBCD：每次选 layer，只保留该 layer 的 local momentum；做 K 次局部更新

算法2的特点是每次选中一个block以后，对这个block做一个局部的循环，即持续训练这个blockK次迭代。
训练完转到下一个block的时候，就把这个block的momentum弃掉。
这样永远只用维持一个block维度的optimizer states，从而减少显存占用。
选block的方式也和算法1一样，用random和循环两种方式。
'''
import time
import argparse
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

import time
import csv
import os

from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs

from train_utils import (
    load_model_and_tokenizer,
    build_layer_blocks, 
    freeze_except, 
    unfreeze_all, 
    momentum_step, 
    pick_block_idx, 
    batch_iterator,
    save_model_and_tokenizer
)
from train_dataset import AlpacaSFTDataset, SFTDataCollator


def main():
    p = argparse.ArgumentParser()

    # model/data
    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--dataset_name", type=str, default="alpaca")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_length", type=int, default=2048)

    # algo2 params
    p.add_argument("--block_select", choices=["cyclic", "random"], default="cyclic")
    p.add_argument("--K", type=int, default=5, help="local steps per outer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.9)

    # train control
    p.add_argument("--total_outer_steps", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")

    # save control
    p.add_argument("--save_dir", type=str, default="runs/algo2")

    args = p.parse_args()

    # ------------------------------------------------------------
    # Prepare model, data, and accelerator
    # ------------------------------------------------------------

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed)

    if accelerator.is_main_process:
        print(f"[Algo2/L-MBCD] select={args.block_select} K={args.K} lr={args.lr} beta={args.beta} mp={args.mixed_precision}")

    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
    
    # dataset
    train_ds = AlpacaSFTDataset(
        tokenizer,
        max_length=args.max_length,
        use_chat_template=True,
    )
    collator = SFTDataCollator(tokenizer, pad_to_multiple_of=8)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    model, train_dl = accelerator.prepare(model, train_dl)
    model.train()

    blocks = build_layer_blocks(model)
    num_blocks = len(blocks)
    if accelerator.is_main_process:
        print(f"[Blocks] num_layers={num_blocks}")

    train_stream = batch_iterator(train_dl)

    # ------------------------------------------------------------
    # Start of Algo2
    # ------------------------------------------------------------

    backward_calls = 0
    start = time.time()

    # log consumption
    csv_f, csv_w = None, None
    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        csv_f = open(os.path.join(args.save_dir, "loss_time.csv"), "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["backward_calls", "loss", "wall_time_sec"])
        csv_f.flush()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for outer_step in range(args.total_outer_steps):
        b = pick_block_idx(outer_step, num_blocks, args.block_select, accelerator)
        freeze_except(blocks, b)

        # local momentum state ONLY for this chosen layer, discarded after outer step
        local_state: Dict[nn.Parameter, torch.Tensor] = {}

        for k in range(args.K):
            batch = next(train_stream)  # fresh batch each local step (closest to your pseudo-code)
            loss = model(**batch).loss
            accelerator.backward(loss)
            backward_calls += 1

            # -- log loss --
            loss_mean = accelerator.gather(loss.detach()).mean().item()
            wall_time = time.time() - start   
            if accelerator.is_main_process:
                csv_w.writerow([backward_calls, loss_mean, wall_time])
                if backward_calls % 50 == 0:
                    csv_f.flush()
            # --- end of log loss ---

            momentum_step(blocks[b], args.lr, args.beta, local_state)
            model.zero_grad(set_to_none=True)

        unfreeze_all(model)

        if outer_step % args.log_every == 0:
            loss_mean = accelerator.gather(loss.detach()).mean().item()
            elapsed = time.time() - start
            max_mem = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
            if accelerator.is_main_process:
                print(f"step={outer_step} block={b} loss={loss_mean:.4f} "
                      f"backward={backward_calls} time={elapsed:.1f}s max_mem={max_mem:.2f}GB")

    save_model_and_tokenizer(accelerator, model, tokenizer, args.save_dir)
    if accelerator.is_main_process:
        print("Done.")
    
    if accelerator.is_main_process and csv_f is not None:
        csv_f.flush()
        csv_f.close()


if __name__ == "__main__":
    main()
