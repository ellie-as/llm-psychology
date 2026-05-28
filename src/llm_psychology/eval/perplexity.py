from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def perplexity_on_text(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str, stride: int = 128) -> Tuple[float, float]:
    """
    Compute negative log-likelihood and perplexity with striding to handle long text.
    Returns (nll, ppl).
    """
    device = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    max_length = model.config.max_position_embeddings
    nlls = []
    for i in range(0, input_ids.size(1), stride):
        begin = max(i + stride - max_length, 0)
        end = min(i + stride, input_ids.size(1))
        trg_len = end - i
        input_ids_slice = input_ids[:, begin:end]
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(input_ids_slice, labels=target_ids)
        nlls.append(out.loss * trg_len)
    nll = torch.stack(nlls).sum() / input_ids.size(1)
    ppl = torch.exp(nll).item()
    return nll.item(), ppl



