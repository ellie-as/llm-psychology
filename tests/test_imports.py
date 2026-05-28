def test_imports():
    import llm_psychology
    from llm_psychology.data.rocstories import load_rocstories
    from llm_psychology.data.statistical_learning import generate_sequence
    from llm_psychology.eval.recall import summarize_recall
    from llm_psychology.models.loaders import load_causal_lm
    from llm_psychology.train import make_character_tokenizer
    from llm_psychology.viz import locate_or_download_final_rel_inf_model, plot_attention_by_layer
    from llm_psychology.xrag import XRAG, xrag_prompt
    assert hasattr(llm_psychology, "__version__")
    assert callable(load_rocstories)
    assert callable(generate_sequence)
    assert callable(summarize_recall)
    assert callable(make_character_tokenizer)
    assert callable(locate_or_download_final_rel_inf_model)
    assert callable(plot_attention_by_layer)
    assert callable(XRAG)
    assert callable(xrag_prompt)
