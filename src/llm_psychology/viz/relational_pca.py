"""PCA visualisation helpers for relational hidden representations.

The defaults mirror the representation extraction used for the relational
inference figures: 50-transition walks, substring/offset matching of entity
names, adjacent-token flanking, L2 normalisation before PCA, and layer-wise PCA
for all-layer spatial plots.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import random
import string
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

FINAL_REL_INF_DRIVE_FOLDER_ID = "13svbXEYla6uB_xpy9Ixacpt3pLkEgvS0"
FINAL_REL_INF_REQUIRED_MODEL_FILES = {
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
FINAL_REL_INF_MODEL_DIRS = {"outputs_graph", "outputs_tree"}


@dataclass(frozen=True)
class RelationalPCAConfig:
    n_samples: int = 1000
    walk_length: int = 50
    layer_index: int = 12
    layers: list[int] = field(default_factory=lambda: list(range(24)))
    batch_size: int = 12
    seed: int = 321
    use_flanking: bool = True
    second_half_only: bool = False
    l2_normalize: bool = True


SpatialExample = tuple[str, list[str], dict[str, tuple[int, int]]]
FamilyExample = tuple[str, list[str], dict[str, int]]


def _model_dir_complete(model_dir: Path, required_files: set[str] = FINAL_REL_INF_REQUIRED_MODEL_FILES) -> bool:
    return all((model_dir / name).exists() for name in required_files)


def find_final_rel_inf_model_root(
    roots: Iterable[str | Path],
    *,
    model_dirs: set[str] = FINAL_REL_INF_MODEL_DIRS,
    required_files: set[str] = FINAL_REL_INF_REQUIRED_MODEL_FILES,
) -> Path:
    """Return the first root containing complete relational-inference model dirs."""
    checked = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        candidates = [root]
        if root.exists():
            candidates.extend(path for path in root.iterdir() if path.is_dir())
        for candidate in candidates:
            checked.append(candidate)
            if all(_model_dir_complete(candidate / dirname, required_files) for dirname in model_dirs):
                return candidate
    searched = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(f"Could not find complete local model folders. Searched:\n{searched}")


def _ensure_gdown():
    if importlib.util.find_spec("gdown") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown>=5"])
    import gdown

    return gdown


def _required_drive_files(gdown, folder_id: str, output_dir: Path):
    files = gdown.download_folder(
        id=folder_id,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
        skip_download=True,
    )
    if not files:
        raise RuntimeError(
            "Could not list the Google Drive folder. Check that the folder is shared "
            "with 'Anyone with the link' and try again."
        )

    selected = []
    for item in files:
        rel_path = Path(item.path)
        if (
            len(rel_path.parts) == 2
            and rel_path.parts[0] in FINAL_REL_INF_MODEL_DIRS
            and rel_path.name in FINAL_REL_INF_REQUIRED_MODEL_FILES
        ):
            selected.append((item.id, rel_path))

    expected = {
        (dirname, filename)
        for dirname in FINAL_REL_INF_MODEL_DIRS
        for filename in FINAL_REL_INF_REQUIRED_MODEL_FILES
    }
    found = {(path.parts[0], path.name) for _file_id, path in selected}
    missing = sorted(expected - found)
    if missing:
        raise RuntimeError(f"The Drive folder is missing required inference files: {missing}")
    return selected


def download_final_rel_inf_model(
    download_dir: str | Path = "../models/final_rel_inf_drive",
    *,
    folder_id: str = FINAL_REL_INF_DRIVE_FOLDER_ID,
) -> Path:
    """Download the required relational-inference model files from Google Drive."""
    download_dir = Path(download_dir).expanduser()
    gdown = _ensure_gdown()
    download_dir.mkdir(parents=True, exist_ok=True)
    selected = _required_drive_files(gdown, folder_id, download_dir)
    for file_id, rel_path in selected:
        out_path = download_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        print(f"Downloading {rel_path}")
        result = gdown.download(
            id=file_id,
            output=str(out_path),
            quiet=False,
            use_cookies=False,
            resume=True,
        )
        if result is None:
            raise RuntimeError(
                f"Google Drive would not allow gdown to download {rel_path} (file id {file_id}). "
                "Check the file permissions or Drive quota for the shared folder."
            )
    return find_final_rel_inf_model_root([download_dir])


def locate_or_download_final_rel_inf_model(
    *,
    model_dir: str | Path | None = None,
    download_dir: str | Path | None = None,
) -> Path:
    """Use a local relational-inference model folder, or download it if needed."""
    roots = []
    model_dir = model_dir or os.environ.get("FINAL_REL_INF_MODEL_DIR")
    if model_dir:
        roots.append(Path(model_dir))
    download_dir = Path(download_dir or os.environ.get("FINAL_REL_INF_DOWNLOAD_DIR", "../models/final_rel_inf_drive"))
    roots.append(download_dir)

    try:
        return find_final_rel_inf_model_root(roots)
    except FileNotFoundError:
        return download_final_rel_inf_model(download_dir)


def _unique_names(rng: random.Random, n: int) -> list[str]:
    names: set[str] = set()
    while len(names) < n:
        names.add("".join(rng.choices(string.ascii_lowercase, k=2)))
    return list(names)


def build_grid_graph(rng: random.Random | None = None) -> tuple[dict[str, list[tuple[str, str]]], list[str], dict[str, tuple[int, int]]]:
    """Build the 3x3 directed grid used by the manuscript PCA code."""
    rng = rng or random.Random()
    nodes = _unique_names(rng, 9)

    east_pairs = [
        (nodes[0], nodes[1]),
        (nodes[1], nodes[2]),
        (nodes[3], nodes[4]),
        (nodes[4], nodes[5]),
        (nodes[6], nodes[7]),
        (nodes[7], nodes[8]),
    ]
    south_pairs = [
        (nodes[0], nodes[3]),
        (nodes[3], nodes[6]),
        (nodes[1], nodes[4]),
        (nodes[4], nodes[7]),
        (nodes[2], nodes[5]),
        (nodes[5], nodes[8]),
    ]
    west_pairs = [(v, u) for u, v in east_pairs]
    north_pairs = [(v, u) for u, v in south_pairs]

    adjacency: dict[str, list[tuple[str, str]]] = {node: [] for node in nodes}
    for source, target in east_pairs:
        adjacency[source].append(("EAST", target))
    for source, target in west_pairs:
        adjacency[source].append(("WEST", target))
    for source, target in south_pairs:
        adjacency[source].append(("SOUTH", target))
    for source, target in north_pairs:
        adjacency[source].append(("NORTH", target))

    grid_positions: dict[str, tuple[int, int]] = {}
    idx = 0
    for row in range(3):
        for col in range(3):
            grid_positions[nodes[idx]] = (row, col)
            idx += 1
    return adjacency, nodes, grid_positions


def build_family_tree(rng: random.Random | None = None) -> tuple[dict[str, list[tuple[str, str]]], list[str], dict[str, int]]:
    """Build the 10-person, three-generation family tree used by the PCA code."""
    rng = rng or random.Random()
    names = _unique_names(rng, 10)
    gp1a, gp1b = names[0], names[1]
    parent1, uncle1 = names[2], names[3]
    gp2a, gp2b = names[4], names[5]
    parent2, aunt2 = names[6], names[7]
    child1, child2 = names[8], names[9]

    relationships = {
        gp1a: {"SPOUSE_OF": [gp1b], "PARENT_OF": [parent1, uncle1]},
        gp1b: {"SPOUSE_OF": [gp1a], "PARENT_OF": [parent1, uncle1]},
        uncle1: {"CHILD_OF": [gp1a, gp1b], "SIBLING_OF": [parent1]},
        gp2a: {"SPOUSE_OF": [gp2b], "PARENT_OF": [parent2, aunt2]},
        gp2b: {"SPOUSE_OF": [gp2a], "PARENT_OF": [parent2, aunt2]},
        aunt2: {"CHILD_OF": [gp2a, gp2b], "SIBLING_OF": [parent2]},
        parent1: {
            "SPOUSE_OF": [parent2],
            "PARENT_OF": [child1, child2],
            "CHILD_OF": [gp1a, gp1b],
            "SIBLING_OF": [uncle1],
        },
        parent2: {
            "SPOUSE_OF": [parent1],
            "PARENT_OF": [child1, child2],
            "CHILD_OF": [gp2a, gp2b],
            "SIBLING_OF": [aunt2],
        },
        child1: {"CHILD_OF": [parent1, parent2], "SIBLING_OF": [child2]},
        child2: {"CHILD_OF": [parent1, parent2], "SIBLING_OF": [child1]},
    }
    for grandparent in [gp1a, gp1b, gp2a, gp2b]:
        relationships[grandparent].setdefault("GRANDPARENT_OF", []).extend([child1, child2])
    for child in [child1, child2]:
        relationships[child].setdefault("GRANDCHILD_OF", []).extend([gp1a, gp1b, gp2a, gp2b])

    adjacency: dict[str, list[tuple[str, str]]] = {person: [] for person in relationships}
    for source, rels in relationships.items():
        for relation, targets in rels.items():
            for target in targets:
                adjacency[source].append((relation, target))

    generation_map = {
        gp1a: 0,
        gp1b: 0,
        gp2a: 0,
        gp2b: 0,
        parent1: 1,
        uncle1: 1,
        parent2: 1,
        aunt2: 1,
        child1: 2,
        child2: 2,
    }
    return adjacency, list(adjacency), generation_map


def generate_random_walk(
    adjacency: dict[str, list[tuple[str, str]]],
    *,
    walk_length: int = 50,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random()
    current = rng.choice(list(adjacency))
    parts = [current]
    for _ in range(walk_length):
        options = adjacency.get(current, [])
        if not options:
            break
        relation, target = rng.choice(options)
        parts.extend([relation, target])
        current = target
    return " ".join(parts)


def make_spatial_examples(
    n_samples: int = 1000,
    *,
    walk_length: int = 50,
    seed: int = 321,
) -> list[SpatialExample]:
    rng = random.Random(seed)
    examples: list[SpatialExample] = []
    for _ in range(n_samples):
        adjacency, node_names, grid_pos = build_grid_graph(rng)
        examples.append((generate_random_walk(adjacency, walk_length=walk_length, rng=rng), node_names, grid_pos))
    return examples


def make_family_examples(
    n_samples: int = 1000,
    *,
    walk_length: int = 50,
    seed: int = 321,
) -> list[FamilyExample]:
    rng = random.Random(seed)
    examples: list[FamilyExample] = []
    for _ in range(n_samples):
        adjacency, node_names, gen_map = build_family_tree(rng)
        examples.append((generate_random_walk(adjacency, walk_length=walk_length, rng=rng), node_names, gen_map))
    return examples


def load_causal_lm_with_tokenizer(model_dir: str | Path, *, device: str = "auto"):
    """Load a local causal LM and fast tokenizer for offset-based extraction."""
    model_dir = Path(model_dir)
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Relational PCA extraction requires a fast tokenizer with offset mappings.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def substring_positions(haystack: str, needle: str) -> list[tuple[int, int]]:
    result = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        result.append((idx, idx + len(needle)))
        start = idx + 1
    return result


def gather_embeddings_for_span(
    offsets: Sequence[tuple[int, int]],
    hidden_states: np.ndarray,
    span: tuple[int, int],
    *,
    flanking: bool = True,
) -> np.ndarray | None:
    start_needed, end_needed = span
    entity_idxs = [
        i
        for i, (start, end) in enumerate(offsets)
        if not (end <= start_needed or start >= end_needed)
    ]
    if not entity_idxs:
        return None
    if flanking:
        all_idxs = set(entity_idxs)
        min_idx, max_idx = min(entity_idxs), max(entity_idxs)
        if min_idx > 0:
            all_idxs.add(min_idx - 1)
        if max_idx < len(offsets) - 1:
            all_idxs.add(max_idx + 1)
        vecs = [hidden_states[i] for i in sorted(all_idxs)]
    else:
        vecs = [hidden_states[i] for i in entity_idxs]
    return np.mean(vecs, axis=0)


def average_locations_via_substring(
    prompt: str,
    offsets: Sequence[tuple[int, int]],
    hidden_states: np.ndarray,
    locs: Iterable[str],
    *,
    flanking: bool = True,
    second_half_only: bool = False,
) -> dict[str, np.ndarray]:
    half = len(prompt) // 2 if second_half_only else 0
    loc_means = {}
    for loc in locs:
        pos_list = substring_positions(prompt, loc)
        if second_half_only:
            pos_list = [(start, end) for start, end in pos_list if start >= half]
        vecs = [
            value
            for start, end in pos_list
            for value in [gather_embeddings_for_span(offsets, hidden_states, (start, end), flanking=flanking)]
            if value is not None
        ]
        if vecs:
            loc_means[loc] = np.mean(vecs, axis=0)
    return loc_means


def _batch_hidden_states(model, tokenizer, prompts: list[str], layers: list[int]):
    device = next(model.parameters()).device
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    )
    offsets_mapping = enc.pop("offset_mapping").cpu().numpy().tolist()
    attention_mask = enc["attention_mask"].cpu().numpy()
    model_inputs = {key: value.to(device) for key, value in enc.items()}
    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
    hidden_by_layer = [outputs.hidden_states[layer].detach().cpu().numpy() for layer in layers]
    valid_offsets = []
    for row, mask in enumerate(attention_mask):
        n_valid = int(mask.sum())
        valid_offsets.append([tuple(pair) for pair in offsets_mapping[row][:n_valid]])
    return hidden_by_layer, valid_offsets


def collect_spatial_points_by_layer(
    model,
    tokenizer,
    examples: Sequence[SpatialExample],
    *,
    layers: Sequence[int] | None = None,
    config: RelationalPCAConfig | None = None,
) -> dict[int, list[dict]]:
    config = config or RelationalPCAConfig()
    layers = list(config.layers if layers is None else layers)
    return _collect_points_by_layer(model, tokenizer, examples, layers=layers, config=config, kind="spatial")


def collect_family_points_by_layer(
    model,
    tokenizer,
    examples: Sequence[FamilyExample],
    *,
    layers: Sequence[int] | None = None,
    config: RelationalPCAConfig | None = None,
) -> dict[int, list[dict]]:
    config = config or RelationalPCAConfig()
    layers = list([config.layer_index] if layers is None else layers)
    return _collect_points_by_layer(model, tokenizer, examples, layers=layers, config=config, kind="family")


def _collect_points_by_layer(
    model,
    tokenizer,
    examples: Sequence,
    *,
    layers: list[int],
    config: RelationalPCAConfig,
    kind: str,
) -> dict[int, list[dict]]:
    points_by_layer: dict[int, list[dict]] = defaultdict(list)
    for start in range(0, len(examples), config.batch_size):
        batch = examples[start : start + config.batch_size]
        prompts = [item[0] for item in batch]
        hidden_by_layer, offsets = _batch_hidden_states(model, tokenizer, prompts, layers)

        for batch_idx, (prompt, node_names, label_map) in enumerate(batch):
            for layer_pos, layer_idx in enumerate(layers):
                hidden = hidden_by_layer[layer_pos][batch_idx, : len(offsets[batch_idx]), :]
                loc_repr = average_locations_via_substring(
                    prompt,
                    offsets[batch_idx],
                    hidden,
                    node_names,
                    flanking=config.use_flanking,
                    second_half_only=config.second_half_only,
                )
                for node_name in node_names:
                    if node_name not in loc_repr:
                        continue
                    if kind == "spatial":
                        row, col = label_map[node_name]
                        points_by_layer[layer_idx].append(
                            {
                                "vector": loc_repr[node_name],
                                "grid_position": (int(row), int(col)),
                                "run_idx": start + batch_idx,
                            }
                        )
                    else:
                        points_by_layer[layer_idx].append(
                            {
                                "vector": loc_repr[node_name],
                                "generation": int(label_map[node_name]),
                                "run_idx": start + batch_idx,
                            }
                        )
        del hidden_by_layer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return dict(points_by_layer)


def pca_2d(vectors: np.ndarray, *, l2_normalize: bool = True) -> np.ndarray:
    """Fit paper-style PCA: optional L2 normalisation, mean centring, 2 PCs."""
    if len(vectors) < 3:
        return np.zeros((len(vectors), 2))
    X = vectors.astype(np.float64, copy=True)
    if l2_normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
    X = X - X.mean(axis=0, keepdims=True)
    n_components = min(2, X.shape[0], X.shape[1])
    result = PCA(n_components=n_components, random_state=42).fit_transform(X)
    if result.shape[1] < 2:
        result = np.column_stack([result, np.zeros(len(result))])
    return result


def compact_points_with_pca(points: Sequence[dict], *, l2_normalize: bool = True) -> list[dict]:
    """Replace vectors with `pc1`/`pc2` coordinates while keeping metadata."""
    if not points:
        return []
    if "pc1" in points[0] and "pc2" in points[0] and "vector" not in points[0]:
        return list(points)
    coords = pca_2d(np.asarray([point["vector"] for point in points], dtype=np.float32), l2_normalize=l2_normalize)
    compact = []
    for point, coord in zip(points, coords):
        row = {key: value for key, value in point.items() if key not in {"vector", "pc1", "pc2"}}
        row["pc1"] = float(coord[0])
        row["pc2"] = float(coord[1])
        compact.append(row)
    return compact


def plot_spatial_layer_pca(
    points: Sequence[dict],
    *,
    title: str = "Spatial entity PCA",
    l2_normalize: bool = True,
    ax=None,
):
    compact = compact_points_with_pca(points, l2_normalize=l2_normalize)
    fig, ax = _ensure_axes(ax, figsize=(5, 5))
    grid_positions = sorted({tuple(point["grid_position"]) for point in compact})
    cmap = plt.get_cmap("tab10")
    pos_to_color = {pos: cmap(i % 10) for i, pos in enumerate(grid_positions)}
    for point in compact:
        pos = tuple(point["grid_position"])
        ax.scatter(point["pc1"], point["pc2"], color=pos_to_color[pos], alpha=0.3, s=8)
    for pos in grid_positions:
        ax.scatter([], [], color=pos_to_color[pos], label=str(pos), s=20)
    ax.legend(fontsize=7, ncol=3, loc="upper left", markerscale=1.0, handletextpad=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    return fig


def plot_family_layer_pca(
    points: Sequence[dict],
    *,
    title: str = "Family entity PCA",
    generation_labels: dict[int, str] | None = None,
    l2_normalize: bool = True,
    ax=None,
):
    compact = compact_points_with_pca(points, l2_normalize=l2_normalize)
    fig, ax = _ensure_axes(ax, figsize=(5, 5))
    generation_labels = generation_labels or {0: "Grandparent", 1: "Parent", 2: "Child"}
    generations = sorted({int(point["generation"]) for point in compact})
    cmap = plt.get_cmap("tab10")
    gen_to_color = {gen: cmap(i % 10) for i, gen in enumerate(generations)}
    for point in compact:
        gen = int(point["generation"])
        ax.scatter(point["pc1"], point["pc2"], color=gen_to_color[gen], alpha=0.3, s=8)
    for gen in generations:
        ax.scatter([], [], color=gen_to_color[gen], label=generation_labels.get(gen, f"Gen {gen}"), s=20)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    return fig


def plot_spatial_all_layers_pca(
    points_by_layer: dict[int, Sequence[dict]],
    *,
    l2_normalize: bool = True,
    scatter_alpha: float = 0.3,
    scatter_size: float = 2,
):
    rows = []
    for layer_idx in sorted(points_by_layer):
        compact = compact_points_with_pca(points_by_layer[layer_idx], l2_normalize=l2_normalize)
        for point in compact:
            row, col = point["grid_position"]
            rows.append(
                {
                    "layer": int(layer_idx),
                    "pc1": point["pc1"],
                    "pc2": point["pc2"],
                    "grid_position": (int(row), int(col)),
                }
            )

    layers = sorted({row["layer"] for row in rows})
    grid_positions = sorted({row["grid_position"] for row in rows})
    cmap = plt.get_cmap("tab10")
    pos_to_color = {pos: cmap(i % 10) for i, pos in enumerate(grid_positions)}
    by_layer: dict[int, list[dict]] = {layer: [] for layer in layers}
    for row in rows:
        by_layer[row["layer"]].append(row)

    n_cols = 6 if len(layers) >= 12 else max(1, min(3, len(layers)))
    n_rows = int(np.ceil(len(layers) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.35, n_rows * 1.35), squeeze=False)
    for ax, layer in zip(axes.ravel(), layers):
        layer_rows = by_layer[layer]
        ax.scatter(
            [float(row["pc1"]) for row in layer_rows],
            [float(row["pc2"]) for row in layer_rows],
            color=[pos_to_color[row["grid_position"]] for row in layer_rows],
            alpha=scatter_alpha,
            s=scatter_size,
        )
        ax.set_title(f"Layer {layer}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
    for ax in axes.ravel()[len(layers) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=np.sqrt(20), color=pos_to_color[pos], label=str(pos))
        for pos in grid_positions
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(grid_positions), fontsize=5, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.08, wspace=0.08, hspace=0.22)
    return fig


def _ensure_axes(ax, *, figsize: tuple[int, int]):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
