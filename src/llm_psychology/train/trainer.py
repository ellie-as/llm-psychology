from dataclasses import dataclass
from typing import Iterable, List, Optional

from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling,
                          PreTrainedTokenizerBase)


@dataclass
class LMDatasetConfig:
    train_path: str
    eval_path: Optional[str] = None
    block_size: int = 256


def _chunk(values: List[int], block_size: int) -> List[List[int]]:
    if not values:
        return []
    if len(values) <= block_size:
        return [values]
    total_length = (len(values) // block_size) * block_size
    return [values[i:i + block_size] for i in range(0, total_length, block_size)]


def build_text_dataset(tokenizer: PreTrainedTokenizerBase, cfg: LMDatasetConfig):
    data_files = {"train": cfg.train_path, "validation": cfg.eval_path} if cfg.eval_path else {"train": cfg.train_path}
    dataset = load_dataset("text", data_files=data_files)

    def tokenize(batch):
        return tokenizer(batch["text"])  # do not truncate here; group later

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])  # type: ignore

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        result = {k: _chunk(t, cfg.block_size) for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(group_texts, batched=True)
    return lm_datasets


def _default_lora_target_modules(model) -> List[str]:
    known = {
        "c_attn",
        "c_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "query",
        "key",
        "value",
        "dense",
    }
    found = set()
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", maxsplit=1)[-1]
        if leaf in known:
            found.add(leaf)
    return sorted(found) or ["c_attn"]


def _apply_peft(
    model,
    method: Optional[str],
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Optional[Iterable[str]],
):
    if method is None:
        return model
    method = method.lower()
    if method not in {"lora", "peft"}:
        raise ValueError("peft_method must be one of None, 'lora', or 'peft'.")
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Install `peft` to train with LoRA/PEFT methods.") from exc

    target_modules = list(lora_target_modules) if lora_target_modules is not None else _default_lora_target_modules(model)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    return get_peft_model(model, config)


def train_causal_lm(model_name: str,
                    train_path: str,
                    eval_path: Optional[str] = None,
                    output_dir: str = "./outputs",
                    tokenizer: Optional[PreTrainedTokenizerBase] = None,
                    tokenizer_name_or_path: Optional[str] = None,
                    num_train_epochs: int = 1,
                    max_steps: Optional[int] = None,
                    block_size: int = 256,
                    per_device_train_batch_size: int = 2,
                    per_device_eval_batch_size: int = 2,
                    learning_rate: float = 5e-5,
                    warmup_steps: int = 0,
                    weight_decay: float = 0.0,
                    logging_steps: int = 10,
                    save_steps: int = 1000,
                    eval_steps: int = 200,
                    fp16: bool = False,
                    from_scratch: bool = False,
                    vocab_size: Optional[int] = None,
                    n_layer: int = 2,
                    n_head: int = 2,
                    n_embd: int = 128,
                    peft_method: Optional[str] = None,
                    lora_r: int = 8,
                    lora_alpha: int = 16,
                    lora_dropout: float = 0.05,
                    lora_target_modules: Optional[Iterable[str]] = None,
                    seed: int = 42):
    """
    Train or fine-tune a causal language model on one or two plain text files.

    Set ``from_scratch=True`` for end-to-end training from a GPT-2 style config.
    Set ``peft_method="lora"`` to train LoRA adapters on top of ``model_name``.
    Pass ``tokenizer`` or ``tokenizer_name_or_path`` to train from text with a
    custom tokenizer, such as a character-level tokenizer.
    The return value remains ``(model, tokenizer, metrics)`` for compatibility.
    """
    if tokenizer is not None and tokenizer_name_or_path is not None:
        raise ValueError("Pass either tokenizer or tokenizer_name_or_path, not both.")
    if tokenizer is None:
        tokenizer_source = tokenizer_name_or_path or (model_name if not from_scratch else "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if from_scratch:
        effective_vocab_size = max(vocab_size or len(tokenizer), len(tokenizer))
        config = GPT2Config(
            vocab_size=effective_vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_positions=block_size,
            n_ctx=block_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = GPT2LMHeadModel(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, low_cpu_mem_usage=True)
    model = _apply_peft(
        model,
        peft_method,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )

    cfg = LMDatasetConfig(train_path=train_path, eval_path=eval_path, block_size=block_size)
    lm_datasets = build_text_dataset(tokenizer, cfg)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps" if eval_path else "no",
        eval_steps=eval_steps,
        fp16=fp16,
        report_to=[],
        save_total_limit=2,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    metrics = {}
    if eval_path:
        metrics = trainer.evaluate()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer, metrics
