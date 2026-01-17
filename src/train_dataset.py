from typing import Optional, Dict, Any, List
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from dataclasses import dataclass
from transformers import PreTrainedTokenizer

IGNORE_INDEX = -100

def _alpaca_prompt(instruction: str, inp: str) -> str:
    instruction = instruction.strip()
    inp = (inp or "").strip()
    if inp:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n"
        )

class AlpacaSFTDataset(Dataset):
    """
    For CausalLM SFT:
      - Build prompt and full text
      - Tokenize full text
      - Mask labels for prompt tokens (set to IGNORE_INDEX)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        system_prompt: Optional[str] = None,
        use_chat_template: bool = True,
    ):
        dataset = load_dataset("tatsu-lab/alpaca")
        self.ds = dataset['train']
        self.tok = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template and getattr(tokenizer, "chat_template", None)

        # pad_token for llama/gpt style
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __len__(self):
        return len(self.ds)

    def _build_texts(self, ex: Dict[str, Any]):
        instruction = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        output = (ex.get("output") or "").strip()

        if self.use_chat_template:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            user_content = instruction if not inp else f"{instruction}\n\n{inp}"
            messages.append({"role": "user", "content": user_content})

            # prompt text ends with an assistant header (no content)
            prompt_text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # full text includes assistant answer
            full_messages = messages + [{"role": "assistant", "content": output}]
            full_text = self.tok.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            return prompt_text, full_text
        else:
            prompt_text = _alpaca_prompt(instruction, inp)
            full_text = prompt_text + output
            return prompt_text, full_text

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.ds[idx]
        prompt_text, full_text = self._build_texts(ex)

        # tokenize prompt to get prompt_len
        prompt_ids = self.tok(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        enc = self.tok(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        # labels: ignore prompt part
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = IGNORE_INDEX

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class SFTDataCollator:
    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = 8
    ignore_index: int = IGNORE_INDEX

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Extract labels before padding (tokenizer.pad doesn't handle labels properly)
        labels = [feature.pop("labels") for feature in features] if "labels" in features[0] else None
        
        # pad input_ids & attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Manually pad labels to match input_ids length
        if labels is not None:
            max_label_length = batch["input_ids"].shape[1]
            padded_labels = []
            for label in labels:
                padding_length = max_label_length - len(label)
                if padding_length > 0:
                    # Pad with ignore_index
                    padded_label = torch.cat([
                        label,
                        torch.full((padding_length,), self.ignore_index, dtype=label.dtype)
                    ])
                else:
                    padded_label = label[:max_label_length]
                padded_labels.append(padded_label)
            
            batch["labels"] = torch.stack(padded_labels)
            
            # Double-check: ensure attention_mask padding positions have ignore_index
            batch["labels"][batch["attention_mask"] == 0] = self.ignore_index
        
        return batch



if __name__ == "__main__":
    dataset = load_dataset("tatsu-lab/alpaca")
    print(dataset['train'][0])