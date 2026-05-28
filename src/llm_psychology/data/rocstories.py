from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

ROC_SENTENCE_COLUMNS = [f"sentence{i}" for i in range(1, 6)]
PACKAGED_ROCSTORIES_FILENAME = "stories_train.csv"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_rocstories_paths() -> List[Path]:
    candidates: List[Path] = []
    env_path = os.environ.get("ROCSTORIES_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path(__file__).with_name(PACKAGED_ROCSTORIES_FILENAME))
    root = _repo_root()
    candidates.extend(
        [
            root / "data" / "stories_train.csv",
            root / "data" / "rocstories" / "stories_train.csv",
            Path.cwd() / "data" / "stories_train.csv",
            Path.cwd() / "data" / "rocstories" / "stories_train.csv",
        ]
    )
    return candidates


def find_rocstories_path(path: Optional[Union[str, Path]] = None) -> Path:
    if path is not None:
        candidate = Path(path).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"ROCStories file not found: {candidate}")
    for candidate in default_rocstories_paths():
        if candidate.exists():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in default_rocstories_paths())
    raise FileNotFoundError(
        "Could not find ROCStories CSV. The package normally includes "
        f"`llm_psychology.data/{PACKAGED_ROCSTORIES_FILENAME}`. You can also set "
        "ROCSTORIES_PATH or place the file at `data/stories_train.csv` in the repository.\n"
        "Searched:\n" + searched
    )


def _story_text(row: pd.Series, sentence_columns: Sequence[str] = ROC_SENTENCE_COLUMNS) -> str:
    return " ".join(str(row[col]).strip() for col in sentence_columns if pd.notna(row[col]) and str(row[col]).strip())


def load_rocstories(
    path: Optional[Union[str, Path]] = None,
    *,
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Load a ROCStories CSV and return normalized columns.

    The official ROCStories format has `sentence1` through `sentence5`.
    The returned dataframe includes `text`, `cue`, `continuation`, and `ending`.
    """
    csv_path = find_rocstories_path(path)
    df = pd.read_csv(csv_path)
    missing = [column for column in ROC_SENTENCE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"ROCStories CSV is missing required columns: {missing}")
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if limit is not None:
        df = df.head(limit).copy()
    df = df.copy()
    df["text"] = df.apply(_story_text, axis=1)
    df["cue"] = df["sentence1"].astype(str)
    df["continuation"] = df[[f"sentence{i}" for i in range(2, 6)]].astype(str).agg(" ".join, axis=1)
    df["ending"] = df["sentence5"].astype(str)
    return df


def write_story_text_files(
    stories: pd.DataFrame,
    train_path: Union[str, Path],
    eval_path: Optional[Union[str, Path]] = None,
    *,
    eval_fraction: float = 0.1,
    seed: int = 0,
    text_column: str = "text",
) -> Tuple[Path, Optional[Path]]:
    if text_column not in stories.columns:
        raise ValueError(f"Missing text column: {text_column}")
    train_path = Path(train_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_out = Path(eval_path) if eval_path is not None else None
    if eval_out is not None:
        eval_out.parent.mkdir(parents=True, exist_ok=True)

    shuffled = stories.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    eval_count = int(round(len(shuffled) * eval_fraction)) if eval_out is not None else 0
    eval_count = min(max(eval_count, 1 if eval_out is not None and len(shuffled) > 1 else 0), max(len(shuffled) - 1, 0))
    eval_df = shuffled.head(eval_count)
    train_df = shuffled.iloc[eval_count:]
    train_path.write_text("\n".join(train_df[text_column].astype(str)) + "\n")
    if eval_out is not None:
        eval_out.write_text("\n".join(eval_df[text_column].astype(str)) + "\n")
    return train_path, eval_out


def episodic_recall_items(
    stories: pd.DataFrame,
    *,
    cue_sentences: int = 1,
    limit: Optional[int] = None,
) -> List[dict]:
    if not 1 <= cue_sentences < 5:
        raise ValueError("cue_sentences must be between 1 and 4.")
    rows = stories.head(limit) if limit is not None else stories
    items: List[dict] = []
    for _, row in rows.iterrows():
        sentences = [str(row[column]) for column in ROC_SENTENCE_COLUMNS]
        cue = " ".join(sentences[:cue_sentences])
        target = " ".join(sentences[cue_sentences:])
        items.append(
            {
                "cue": cue,
                "target": target,
                "text": " ".join(sentences),
                "kind": "episodic",
                "storyid": row.get("storyid", ""),
                "storytitle": row.get("storytitle", ""),
            }
        )
    return items


def story_cloze_items(stories: pd.DataFrame, *, limit: Optional[int] = None) -> List[dict]:
    rows = stories.head(limit) if limit is not None else stories
    items: List[dict] = []
    for _, row in rows.iterrows():
        cue = " ".join(str(row[column]) for column in ROC_SENTENCE_COLUMNS[:4])
        target = str(row["sentence5"])
        items.append(
            {
                "cue": cue,
                "target": target,
                "text": str(row["text"]),
                "kind": "semantic",
                "storyid": row.get("storyid", ""),
                "storytitle": row.get("storytitle", ""),
            }
        )
    return items


def story_fact_texts(stories: pd.DataFrame, *, limit: Optional[int] = None) -> List[str]:
    rows = stories.head(limit) if limit is not None else stories
    facts: List[str] = []
    for _, row in rows.iterrows():
        prefix = f"Story {row.get('storyid', '')}".strip()
        title = str(row.get("storytitle", "")).strip()
        if title:
            facts.append(f"{prefix} is titled {title}.")
        for index, column in enumerate(ROC_SENTENCE_COLUMNS, start=1):
            facts.append(f"{prefix} sentence {index}: {row[column]}")
    return facts
