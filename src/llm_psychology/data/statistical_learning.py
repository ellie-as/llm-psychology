from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

Token = Union[int, str]


TRANSITION_STRUCTURE: Mapping[Tuple[int, int], int] = {
    (1, 1): 4,
    (1, 2): 3,
    (1, 3): 2,
    (1, 4): 1,
    (1, 5): 5,
    (2, 1): 5,
    (2, 2): 4,
    (2, 3): 3,
    (2, 4): 2,
    (2, 5): 1,
    (3, 1): 3,
    (3, 2): 2,
    (3, 3): 1,
    (3, 4): 5,
    (3, 5): 4,
    (4, 1): 1,
    (4, 2): 5,
    (4, 3): 4,
    (4, 4): 3,
    (4, 5): 2,
    (5, 1): 2,
    (5, 2): 1,
    (5, 3): 5,
    (5, 4): 4,
    (5, 5): 3,
}


def parse_sequence(sequence: Union[str, Sequence[Token]]) -> List[str]:
    if isinstance(sequence, str):
        compact = sequence.strip()
        if compact and all(character in "12345" for character in compact):
            return list(compact)
        return [part for part in sequence.replace(",", " ").split() if part]
    return [str(item) for item in sequence]


def expected_next(previous_two: Sequence[Token]) -> str:
    if len(previous_two) != 2:
        raise ValueError("previous_two must contain exactly two items.")
    key = (int(previous_two[0]), int(previous_two[1]))
    return str(TRANSITION_STRUCTURE[key])


def generate_sequence(
    length: int = 52,
    *,
    error_rate: float = 0.1,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Generate the Durrant-style statistical-learning stream.

    The next item is determined by the previous two items except on
    ``error_rate`` trials, where a random item from 1..5 is emitted.
    """
    if length < 3:
        raise ValueError("length must be at least 3.")
    rng = rng or random.Random()
    sequence = [rng.randint(1, 5), rng.randint(1, 5)]
    while len(sequence) < length:
        if rng.random() > error_rate:
            next_value = TRANSITION_STRUCTURE[(sequence[-2], sequence[-1])]
        else:
            next_value = rng.randint(1, 5)
        sequence.append(next_value)
    return [str(item) for item in sequence]


def sequence_to_text(sequence: Sequence[Token], separator: str = " ") -> str:
    return separator.join(str(item) for item in sequence)


def make_corpus(
    path: Union[str, Path],
    *,
    num_sequences: int = 1000,
    length: int = 52,
    error_rate: float = 0.1,
    seed: int = 0,
    separator: str = " ",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with path.open("w") as handle:
        for _ in range(num_sequences):
            handle.write(sequence_to_text(generate_sequence(length, error_rate=error_rate, rng=rng), separator) + "\n")
    return path


def iter_bigram_examples(sequences: Iterable[Union[str, Sequence[Token]]]):
    for sequence in sequences:
        tokens = parse_sequence(sequence)
        for index in range(len(tokens) - 2):
            yield tokens[index:index + 2], tokens[index + 2]


def transition_probability_matrix(
    sequences: Iterable[Union[str, Sequence[Token]]],
    *,
    tokens: Sequence[Token] = ("1", "2", "3", "4", "5"),
) -> Tuple[np.ndarray, List[str], List[str]]:
    token_labels = [str(token) for token in tokens]
    token_to_col = {token: idx for idx, token in enumerate(token_labels)}
    contexts = [(a, b) for a in token_labels for b in token_labels]
    context_to_row = {context: idx for idx, context in enumerate(contexts)}
    matrix = np.zeros((len(contexts), len(token_labels)), dtype=np.float64)
    for context, target in iter_bigram_examples(sequences):
        context_tuple = (context[0], context[1])
        if context_tuple in context_to_row and target in token_to_col:
            matrix[context_to_row[context_tuple], token_to_col[target]] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    return matrix, [f"{a},{b}" for a, b in contexts], token_labels


def make_attention_probe(
    *,
    context_length: int = 20,
    error_rate: float = 0.0,
    seed: int = 0,
    separator: str = " ",
) -> dict:
    """
    Build a next-item probe where the final answer depends only on the last two items.
    """
    rng = random.Random(seed)
    sequence = generate_sequence(context_length + 1, error_rate=error_rate, rng=rng)
    prompt_tokens = sequence[:-1]
    return {
        "prompt": sequence_to_text(prompt_tokens, separator),
        "target": sequence[-1],
        "sequence": sequence,
        "relevant_item_positions": [len(prompt_tokens) - 2, len(prompt_tokens) - 1],
    }
