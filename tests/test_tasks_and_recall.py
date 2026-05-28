from llm_psychology.data.family_tree import create_extended_family_tree, semantic_qa_pairs
from pathlib import Path

from llm_psychology.data import rocstories
from llm_psychology.data.rocstories import episodic_recall_items as roc_episodic_items
from llm_psychology.data.rocstories import find_rocstories_path, load_rocstories, story_cloze_items, write_story_text_files
from llm_psychology.data.spatial import make_grid_graph, generate_random_walk, is_valid_path
from llm_psychology.data.statistical_learning import expected_next, generate_sequence, make_attention_probe
from llm_psychology.eval.recall import evaluate_recall_predictions, summarize_recall
from llm_psychology.viz import find_final_rel_inf_model_root


def test_statistical_learning_probe_uses_last_two_items():
    probe = make_attention_probe(context_length=10, error_rate=0.0, seed=3)
    prompt_items = probe["prompt"].split()
    assert probe["target"] == expected_next(prompt_items[-2:])
    assert probe["relevant_item_positions"] == [len(prompt_items) - 2, len(prompt_items) - 1]


def test_spatial_walk_validity():
    graph = make_grid_graph(size=3, seed=1)
    walk = generate_random_walk(graph, walk_length=12)
    assert is_valid_path(walk, graph)


def test_family_tree_semantic_questions():
    tree = create_extended_family_tree(seed=2)
    questions = semantic_qa_pairs(tree, max_questions=5)
    assert questions
    assert {"question", "answer", "relation"}.issubset(questions[0])


def test_recall_metrics_summary():
    results = evaluate_recall_predictions(
        predictions=["The answer is AB."],
        targets=["AB"],
        cues=["Who?"],
        kinds=["semantic"],
    )
    summary = summarize_recall(results)
    assert summary["n"] == 1
    assert summary["contains_target"] == 1.0
    assert summary["token_f1"] > 0.0


def test_rocstories_loader_and_items(tmp_path):
    csv_path = tmp_path / "stories_train.csv"
    csv_path.write_text(
        "storyid,storytitle,sentence1,sentence2,sentence3,sentence4,sentence5\n"
        "s1,Title One,One.,Two.,Three.,Four.,Five.\n"
        "s2,Title Two,A.,B.,C.,D.,E.\n"
    )
    stories = load_rocstories(csv_path)
    assert stories.loc[0, "text"] == "One. Two. Three. Four. Five."
    assert roc_episodic_items(stories, cue_sentences=1, limit=1)[0]["target"] == "Two. Three. Four. Five."
    assert story_cloze_items(stories, limit=1)[0]["target"] == "Five."

    train_path, val_path = write_story_text_files(
        stories,
        tmp_path / "train.txt",
        tmp_path / "val.txt",
        eval_fraction=0.5,
    )
    assert train_path.exists()
    assert val_path is not None and val_path.exists()


def test_rocstories_packaged_csv_is_default(monkeypatch):
    monkeypatch.delenv("ROCSTORIES_PATH", raising=False)
    packaged_path = Path(rocstories.__file__).with_name(rocstories.PACKAGED_ROCSTORIES_FILENAME)
    assert find_rocstories_path() == packaged_path
    stories = load_rocstories(limit=2)
    assert len(stories) == 2
    assert {"text", "cue", "continuation", "ending"}.issubset(stories.columns)


def test_find_final_rel_inf_model_root(tmp_path):
    required_files = {
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    }
    for dirname in ["outputs_graph", "outputs_tree"]:
        model_dir = tmp_path / dirname
        model_dir.mkdir()
        for filename in required_files:
            (model_dir / filename).write_text("{}")

    assert find_final_rel_inf_model_root([tmp_path]) == tmp_path.resolve()
