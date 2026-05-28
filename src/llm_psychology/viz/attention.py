from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def attention_matrix(
    attentions,
    *,
    batch_index: int = 0,
    layer: int = -1,
    head: Optional[int] = None,
) -> np.ndarray:
    """
    Select an attention matrix from ``(batch, layers, heads, seq, seq)`` output.
    If ``head`` is omitted, heads are averaged.
    """
    if isinstance(attentions, torch.Tensor):
        array = attentions.detach().cpu().numpy()
    else:
        array = np.asarray(attentions)
    selected = array[batch_index, layer]
    if head is None:
        return selected.mean(axis=0)
    return selected[head]


def plot_attention_heatmap(
    attentions,
    tokens: Sequence[str],
    *,
    batch_index: int = 0,
    layer: int = -1,
    head: Optional[int] = None,
    title: Optional[str] = None,
):
    matrix = attention_matrix(attentions, batch_index=batch_index, layer=layer, head=head)
    fig, ax = plt.subplots(figsize=(max(5, len(tokens) * 0.45), max(4, len(tokens) * 0.38)))
    im = ax.imshow(matrix[: len(tokens), : len(tokens)], cmap="magma", vmin=0)
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Attended token")
    ax.set_ylabel("Query token")
    label = "mean heads" if head is None else f"head {head}"
    ax.set_title(title or f"Attention layer {layer} ({label})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_attention_by_layer(
    attentions,
    tokens: Sequence[str],
    *,
    batch_index: int = 0,
    layers: Optional[Sequence[int]] = None,
    head: Optional[int] = None,
    relevant_positions: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
):
    """
    Plot one attention heatmap per layer.

    If ``head`` is omitted, heads are averaged. ``relevant_positions`` can be
    used to mark columns that correspond to task-relevant attended tokens.
    """
    if isinstance(attentions, torch.Tensor):
        array = attentions.detach().cpu().numpy()
    else:
        array = np.asarray(attentions)

    n_layers = array.shape[1]
    selected_layers = list(range(n_layers)) if layers is None else list(layers)
    resolved_layers = [layer if layer >= 0 else n_layers + layer for layer in selected_layers]

    matrices = [
        attention_matrix(attentions, batch_index=batch_index, layer=layer, head=head)[: len(tokens), : len(tokens)]
        for layer in selected_layers
    ]
    vmax = max(float(matrix.max()) for matrix in matrices) if matrices else 1.0
    n_cols = min(4, max(1, len(matrices)))
    n_rows = int(np.ceil(len(matrices) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * max(3.0, len(tokens) * 0.18), n_rows * max(2.8, len(tokens) * 0.16)),
        squeeze=False,
        constrained_layout=True,
    )

    last_image = None
    for plot_idx, (ax, matrix, layer) in enumerate(zip(axes.ravel(), matrices, resolved_layers)):
        last_image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"Layer {layer}", fontsize=9)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        if plot_idx // n_cols == n_rows - 1:
            ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        else:
            ax.set_xticklabels([])
        if plot_idx % n_cols == 0:
            ax.set_yticklabels(tokens, fontsize=7)
        else:
            ax.set_yticklabels([])
        if relevant_positions is not None:
            for position in relevant_positions:
                ax.axvline(position, color="cyan", lw=1.4, alpha=0.9)

    for ax in axes.ravel()[len(matrices) :]:
        ax.axis("off")

    label = "mean heads" if head is None else f"head {head}"
    fig.suptitle(title or f"Attention by layer ({label})", y=1.02)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    return fig
