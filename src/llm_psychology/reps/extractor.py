from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _attention_tensors(attentions) -> bool:
    return attentions is not None and all(isinstance(attention, torch.Tensor) for attention in attentions)


def _force_eager_attention(model) -> bool:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "_attn_implementation"):
        return False
    if getattr(config, "_attn_implementation", None) == "eager":
        return False
    config._attn_implementation = "eager"
    return True


def extract_hidden_states(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, texts: List[str]) -> Tuple[torch.Tensor, Dict]:
    """
    Returns tensor of shape (num_texts, num_layers, seq_len, hidden_dim)
    and the tokenization outputs.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**batch, output_hidden_states=True)
        # outputs.hidden_states: tuple of (num_layers+1) tensors (including embedding layer)
        # Stack to shape (num_layers+1, batch, seq, hidden)
        hs = torch.stack(outputs.hidden_states, dim=0)
        # move to (batch, layers, seq, hidden)
        hs = hs.permute(1, 0, 2, 3).contiguous()
    return hs.cpu(), batch


def token_labels(tokenizer: AutoTokenizer, input_ids: torch.Tensor) -> List[List[str]]:
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    labels = []
    for row in input_ids.detach().cpu().tolist():
        labels.append(tokenizer.convert_ids_to_tokens(row))
    return labels


def extract_token_representations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    *,
    layers: Optional[Iterable[int]] = None,
    max_length: Optional[int] = None,
    include_pad_tokens: bool = False,
) -> pd.DataFrame:
    """
    Return one row per token per selected layer.

    The ``vector`` column contains a NumPy array for the hidden representation.
    Layers include the embedding layer at index 0.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        ).to(device)
        outputs = model(**batch, output_hidden_states=True)
    hidden_states = torch.stack(outputs.hidden_states, dim=0).permute(1, 0, 2, 3).cpu()
    input_ids = batch["input_ids"].detach().cpu()
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.detach().cpu()
    labels = token_labels(tokenizer, input_ids)
    selected_layers = list(layers) if layers is not None else list(range(hidden_states.shape[1]))

    rows = []
    for text_index, text in enumerate(texts):
        for token_index, token in enumerate(labels[text_index]):
            if not include_pad_tokens and attention_mask is not None and int(attention_mask[text_index, token_index]) == 0:
                continue
            token_id = int(input_ids[text_index, token_index])
            for layer in selected_layers:
                rows.append(
                    {
                        "text_index": text_index,
                        "text": text,
                        "token_index": token_index,
                        "token": token,
                        "token_id": token_id,
                        "layer": int(layer),
                        "vector": hidden_states[text_index, layer, token_index].numpy(),
                    }
                )
    return pd.DataFrame(rows)


def extract_attention_weights(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    *,
    max_length: Optional[int] = None,
) -> Tuple[torch.Tensor, List[List[str]], Dict]:
    """
    Return attentions with shape ``(batch, layers, heads, seq, seq)`` and tokens.
    """
    model.eval()
    device = next(model.parameters()).device
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        outputs = model(**batch, output_attentions=True)
    if not _attention_tensors(outputs.attentions):
        if _force_eager_attention(model):
            with torch.no_grad():
                outputs = model(**batch, output_attentions=True)
    if not _attention_tensors(outputs.attentions):
        raise ValueError(
            "The model did not return attention tensors. Some Transformers models "
            "return None for attentions with the default SDPA backend; set "
            "`model.config._attn_implementation = 'eager'` before extraction."
        )
    attentions = torch.stack(outputs.attentions, dim=0).permute(1, 0, 2, 3, 4).contiguous().cpu()
    return attentions, token_labels(tokenizer, batch["input_ids"]), batch


