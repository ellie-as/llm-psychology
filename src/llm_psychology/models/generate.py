from typing import List, Optional

import torch
from transformers.generation import LogitsProcessor, LogitsProcessorList


def simple_generate(model, tokenizer, prompts: List[str],
                    max_new_tokens: int = 50,
                    temperature: float = 0.8,
                    top_p: float = 0.95,
                    do_sample: bool = True,
                    device: Optional[str] = None) -> List[str]:
    model.eval()
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = model.to(device)

    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        outputs.append(text)
    return outputs


class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: List[int]):
        super().__init__()
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not hasattr(self, "_mask"):
            mask = torch.full_like(scores, float("-inf"))
            mask[:, self.allowed_token_ids] = 0.0
            self._mask = mask
        return scores + self._mask


def constrained_generate(model, tokenizer, prompts: List[str],
                         allowed_tokens: List[str],
                         max_new_tokens: int = 10,
                         do_sample: bool = False,
                         device: Optional[str] = None) -> List[str]:
    model.eval()
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = model.to(device)

    allowed_ids = tokenizer.convert_tokens_to_ids(allowed_tokens)
    logits_processor = LogitsProcessorList([AllowedTokensLogitsProcessor(allowed_ids)])

    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                logits_processor=logits_processor,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        outputs.append(text)
    return outputs



