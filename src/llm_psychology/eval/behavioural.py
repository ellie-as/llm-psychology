from typing import Iterable, List, Tuple

import torch


def next_token_accuracy(model, tokenizer, texts: Iterable[str], positions_per_text: int = 3,
                        max_length: int = 128) -> Tuple[float, List[Tuple[str, int, str, str]]]:
    """
    Compute greedy top-1 next-token accuracy at a few evenly spaced positions in each text.

    Returns (accuracy, details) where details contains tuples of
    (text, position, predicted_token, true_token).
    """
    model.eval()
    device = next(model.parameters()).device
    total = 0
    correct = 0
    details: List[Tuple[str, int, str, str]] = []

    for text in texts:
        tokens = tokenizer.tokenize(text)
        if len(tokens) < 2:
            continue
        # choose evenly spaced positions not including first token
        max_pos = max(1, len(tokens) - 1)
        idxs = list(range(1, max_pos))
        if len(idxs) > positions_per_text:
            step = max(1, len(idxs) // positions_per_text)
            idxs = idxs[::step][:positions_per_text]

        for pos in idxs:
            prefix_tokens = tokens[:pos]
            true_token = tokens[pos]
            prefix = tokenizer.convert_tokens_to_string(prefix_tokens)
            enc = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=max_length)
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc.get("attention_mask", None).to(device) if enc.get("attention_mask") is not None else None,
                    do_sample=False,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            out_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            out_tokens = tokenizer.tokenize(out_text)
            pred_token = out_tokens[pos] if len(out_tokens) > pos else ""
            total += 1
            if pred_token == true_token:
                correct += 1
            details.append((text, pos, pred_token, true_token))

    acc = correct / total if total else 0.0
    return acc, details


