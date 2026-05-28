from pathlib import Path

def test_trainer_smoke(tmp_path: Path):
    from llm_psychology.reps.extractor import extract_attention_weights
    from llm_psychology.train.trainer import train_causal_lm

    # tiny dataset
    train = tmp_path / "t.txt"
    val = tmp_path / "v.txt"
    train.write_text("hello world\n" * 10)
    val.write_text("hello world\n" * 2)

    model, tok, metrics = train_causal_lm(
        model_name="sshleifer/tiny-gpt2",
        train_path=str(train),
        eval_path=str(val),
        output_dir=str(tmp_path / "out"),
        max_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
    )
    assert (tmp_path / "out").exists()
    attentions, tokens, _batch = extract_attention_weights(model, tok, ["hello world"])
    assert attentions.ndim == 5
    assert tokens[0]


def test_character_tokenizer_from_scratch_training(tmp_path: Path):
    from llm_psychology.data.statistical_learning import make_corpus
    from llm_psychology.train.tokenizers import make_character_tokenizer
    from llm_psychology.train.trainer import train_causal_lm

    train = make_corpus(tmp_path / "train.txt", num_sequences=20, length=12, seed=1, separator="")
    val = make_corpus(tmp_path / "val.txt", num_sequences=4, length=12, seed=2, separator="")
    tokenizer = make_character_tokenizer(["1", "2", "3", "4", "5"])

    assert tokenizer.tokenize("45215") == ["4", "5", "2", "1", "5"]

    model, tok, _metrics = train_causal_lm(
        model_name="gpt2",
        tokenizer=tokenizer,
        train_path=str(train),
        eval_path=str(val),
        output_dir=str(tmp_path / "char_model"),
        from_scratch=True,
        n_layer=1,
        n_head=1,
        n_embd=32,
        max_steps=2,
        block_size=32,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
    )

    assert model.config.vocab_size == len(tok)
    assert (tmp_path / "char_model" / "tokenizer.json").exists()
