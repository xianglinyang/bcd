'''
Algo1 / MBCD：全局 momentum state

算法1有两种情况，就是第2行对应的select block的方式。
一种是随机选，一种是循环的选(从第1个block训到最后一个block，循环这个过程)
'''
import time
import argparse
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

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
    p.add_argument("--model_name_or_path", type=str)
    p.add_argument("--dataset_name", type=str, default="alpaca")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_length", type=int, default=2048)

    # algo1 params
    p.add_argument("--block_select", choices=["cyclic", "random"], default="cyclic")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.9)

    # train control
    p.add_argument("--total_outer_steps", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")

    # save control
    p.add_argument("--save_dir", type=str, default="runs/algo1")


    args = p.parse_args()

    # ------------------------------------------------------------
    # Prepare model, data, and accelerator
    # ------------------------------------------------------------

    # important for dynamic freezing
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)

    if accelerator.is_main_process:
        print(f"[Algo1/MBCD] select={args.block_select} lr={args.lr} beta={args.beta} mp={args.mixed_precision}")

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
    # Start of Algo1
    # ------------------------------------------------------------

    global_state: Dict[nn.Parameter, torch.Tensor] = {}  # keep momentum for all layers
    backward_calls = 0
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for outer_step in range(args.total_outer_steps):
        b = pick_block_idx(outer_step, num_blocks, args.block_select, accelerator)
        freeze_except(blocks, b)

        batch = next(train_stream)
        loss = model(**batch).loss
        accelerator.backward(loss)
        backward_calls += 1

        momentum_step(blocks[b], args.lr, args.beta, global_state)

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
    


if __name__ == "__main__":
    main()
