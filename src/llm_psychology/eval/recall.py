from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch


@dataclass
class RecallExample:
    cue: str
    target: str
    kind: str = "episodic"


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens or not target_tokens:
        return 0.0
    common = 0
    remaining = target_tokens.copy()
    for token in pred_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, target: str) -> bool:
    return normalize_text(prediction) == normalize_text(target)


def contains_target(prediction: str, target: str) -> bool:
    normalized_prediction = normalize_text(prediction)
    normalized_target = normalize_text(target)
    return bool(normalized_target) and normalized_target in normalized_prediction


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embedding_similarity(
    predictions: Sequence[str],
    targets: Sequence[str],
    embedding_model=None,
) -> List[Optional[float]]:
    """
    Compute optional sentence-level semantic similarity.

    ``embedding_model`` may be a SentenceTransformer-like object with ``encode``
    or a Hugging Face sentence-transformers model id. When omitted, similarities
    are returned as ``None``.
    """
    if embedding_model is None:
        return [None for _ in predictions]
    if isinstance(embedding_model, str):
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("Install `sentence-transformers` for embedding similarity.") from exc
        embedding_model = SentenceTransformer(embedding_model)
    pred_embeddings = embedding_model.encode(list(predictions), convert_to_numpy=True)
    target_embeddings = embedding_model.encode(list(targets), convert_to_numpy=True)
    return [_cosine_similarity(pred, target) for pred, target in zip(pred_embeddings, target_embeddings)]


def evaluate_recall_predictions(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    cues: Optional[Sequence[str]] = None,
    kinds: Optional[Sequence[str]] = None,
    embedding_model=None,
) -> pd.DataFrame:
    similarities = embedding_similarity(predictions, targets, embedding_model=embedding_model)
    rows = []
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        rows.append(
            {
                "cue": cues[index] if cues is not None else "",
                "target": target,
                "prediction": prediction,
                "kind": kinds[index] if kinds is not None else "recall",
                "exact_match": exact_match(prediction, target),
                "contains_target": contains_target(prediction, target),
                "token_f1": token_f1(prediction, target),
                "embedding_similarity": similarities[index],
            }
        )
    return pd.DataFrame(rows)


def summarize_recall(results: pd.DataFrame) -> dict:
    if results.empty:
        return {"n": 0, "exact_match": 0.0, "contains_target": 0.0, "token_f1": 0.0}
    summary = {
        "n": int(len(results)),
        "exact_match": float(results["exact_match"].mean()),
        "contains_target": float(results["contains_target"].mean()),
        "token_f1": float(results["token_f1"].mean()),
    }
    if "embedding_similarity" in results and results["embedding_similarity"].notna().any():
        summary["embedding_similarity"] = float(results["embedding_similarity"].dropna().mean())
    return summary


def _as_examples(examples: Iterable[Union[RecallExample, dict]], kind: str) -> List[RecallExample]:
    parsed = []
    for example in examples:
        if isinstance(example, RecallExample):
            parsed.append(example)
        else:
            cue = example.get("cue") or example.get("question") or example.get("prompt")
            target = example.get("target") or example.get("answer")
            if cue is None or target is None:
                raise ValueError("Each recall example must provide cue/question/prompt and target/answer.")
            parsed.append(RecallExample(cue=str(cue), target=str(target), kind=str(example.get("kind", kind))))
    return parsed


def generate_recall(
    model,
    tokenizer,
    examples: Iterable[Union[RecallExample, dict]],
    *,
    prompt_template: str = "{cue}",
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float = 0.8,
    stop_at_newline: bool = True,
    device: Optional[str] = None,
) -> Tuple[List[str], List[RecallExample]]:
    parsed = _as_examples(examples, kind="recall")
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model = model.to(device)
    model.eval()
    predictions: List[str] = []
    for example in parsed:
        prompt = prompt_template.format(cue=example.cue, question=example.cue)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        prediction = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        if stop_at_newline:
            prediction = prediction.splitlines()[0].strip()
        predictions.append(prediction)
    return predictions, parsed


def run_episodic_recall(
    model,
    tokenizer,
    episodes: Iterable[Union[RecallExample, dict]],
    *,
    prompt_template: str = "{cue}",
    embedding_model=None,
    **generate_kwargs,
) -> pd.DataFrame:
    predictions, examples = generate_recall(
        model,
        tokenizer,
        _as_examples(episodes, kind="episodic"),
        prompt_template=prompt_template,
        **generate_kwargs,
    )
    return evaluate_recall_predictions(
        predictions,
        [example.target for example in examples],
        cues=[example.cue for example in examples],
        kinds=["episodic" for _ in examples],
        embedding_model=embedding_model,
    )


def run_semantic_recall(
    model,
    tokenizer,
    questions: Iterable[Union[RecallExample, dict]],
    *,
    prompt_template: str = "Question: {question}\nAnswer:",
    embedding_model=None,
    **generate_kwargs,
) -> pd.DataFrame:
    predictions, examples = generate_recall(
        model,
        tokenizer,
        _as_examples(questions, kind="semantic"),
        prompt_template=prompt_template,
        **generate_kwargs,
    )
    return evaluate_recall_predictions(
        predictions,
        [example.target for example in examples],
        cues=[example.cue for example in examples],
        kinds=["semantic" for _ in examples],
        embedding_model=embedding_model,
    )

