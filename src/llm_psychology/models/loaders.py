from typing import Tuple, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(model_name: str = "gpt2",
                   dtype: Optional[str] = None,
                   device_map: Optional[str] = None,
                   trust_remote_code: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a Hugging Face causal LM and tokenizer with light config.

    Parameters
    ----------
    model_name: str
        HF model id or local path. Defaults to "gpt2" (small CPU friendly).
    dtype: Optional[str]
        e.g., "float16". If provided, forwarded to from_pretrained via torch_dtype.
    device_map: Optional[str]
        e.g., "auto" or "cpu". Defaults to None to respect HF/accelerate env.
    trust_remote_code: bool
        Whether to allow custom model code.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        import torch
        kwargs["torch_dtype"] = getattr(torch, dtype)
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer



