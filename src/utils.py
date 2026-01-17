# ------------------------------------------------------------
# Model and tokenizer loader for BCD
# ------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    return model, tokenizer 



from typing import List
import torch.nn as nn
from accelerate import Accelerator
import torch
from typing import Iterable
import math
from torch.utils.data import DataLoader

# -----------------------------
# Blocks: one layer = one block
# -----------------------------
def build_layer_blocks(model: nn.Module) -> List[List[nn.Parameter]]:
    core = model.module if hasattr(model, "module") else model

    if hasattr(core, "model") and hasattr(core.model, "layers"):
        layers = list(core.model.layers)
        blocks = [[p for p in layer.parameters() if p.requires_grad] for layer in layers]
        if not blocks:
            raise ValueError("Found layers but no trainable params.")
        return blocks

    if hasattr(core, "transformer") and hasattr(core.transformer, "h"):
        layers = list(core.transformer.h)
        blocks = [[p for p in layer.parameters() if p.requires_grad] for layer in layers]
        if not blocks:
            raise ValueError("Found layers but no trainable params.")
        return blocks

    raise ValueError("Unknown CausalLM layout: cannot find layers.")


def freeze_except(blocks, chosen_idx: int):
    for i, params in enumerate(blocks):
        req = (i == chosen_idx)
        for p in params:
            p.requires_grad_(req)

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad_(True)

def zero_grads(params: Iterable[nn.Parameter]):
    for p in params:
        p.grad = None

# ----------------------------
# Sync random block across GPUs
# ----------------------------
def pick_block_idx(step: int, num_blocks: int, mode: str, accelerator: Accelerator) -> int:
    '''
    Pick a block index across GPUs.
    args:
        - step: the current step, used for cyclic selection
        - num_blocks: the number of blocks
        - mode: the mode of block selection, "cyclic" or "random"
        - accelerator: the accelerator
    returns:
        - the chosen block index
    '''
    if mode == "cyclic":
        return step % num_blocks

    # random but synced
    if accelerator.is_main_process:
        idx = torch.randint(0, num_blocks, (1,), device=accelerator.device)
    else:
        idx = torch.zeros(1, dtype=torch.long, device=accelerator.device)

    if accelerator.distributed_type != "NO":
        torch.distributed.broadcast(idx, src=0)
    return int(idx.item())

# ----------------------------
# Momentum update (SGD+Momentum form)
# ----------------------------
@torch.no_grad()
def momentum_step(params, lr: float, beta: float, state: dict):
    # state: param -> momentum buffer
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        buf = state.get(p)
        if buf is None:
            buf = torch.zeros_like(p)
        buf.mul_(beta).add_(g)
        p.add_(buf, alpha=-lr)
        state[p] = buf


# ----------------------------
# Loss wrapper
# ----------------------------
def compute_loss(model: nn.Module, batch: dict):
    """
    Expect HF-style batch: input_ids, attention_mask, labels
    Model returns .loss
    """
    out = model(**batch)
    return out.loss


# ----------------------------
# Data iterator helper (gives new batch each call)
# Replace with your own dataloader
# ----------------------------
def batch_iterator(dataloader):
    it = iter(dataloader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(dataloader)