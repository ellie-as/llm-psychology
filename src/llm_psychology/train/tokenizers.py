from __future__ import annotations

from collections.abc import Sequence

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast


def make_character_tokenizer(
    characters: Sequence[str],
    *,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
    eos_token: str = "<eos>",
) -> PreTrainedTokenizerFast:
    """Create a fast tokenizer that isolates each character as one token."""
    special_tokens = [pad_token, unk_token, eos_token]
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for character in characters:
        if len(character) != 1:
            raise ValueError("Character tokenizer entries must be single-character strings.")
        if character not in vocab:
            vocab[character] = len(vocab)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_token))
    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=pad_token,
        unk_token=unk_token,
        eos_token=eos_token,
    )
