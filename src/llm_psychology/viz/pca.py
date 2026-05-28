from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def pca_by_layer(hidden_states: np.ndarray,
                 layer_indices: Optional[List[int]] = None,
                 color_values: Optional[List[float]] = None,
                 title_prefix: str = "PCA"):
    """
    Plot PCA(2D) per selected layer. hidden_states shape: (batch, layers, seq, hidden)
    """
    batch, layers, seq, hidden = hidden_states.shape
    layer_indices = layer_indices or list(range(layers))

    num_plots = len(layer_indices)
    cols = min(3, num_plots)
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, layer in enumerate(layer_indices):
        reps = hidden_states[:, layer, :, :].reshape(batch * seq, hidden)
        pca = PCA(n_components=2)
        pts = pca.fit_transform(reps)
        ax = axes[i]
        c = color_values if color_values is not None else np.arange(len(pts))
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=c, s=6, cmap="viridis")
        ax.set_title(f"{title_prefix} L{layer}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if color_values is not None:
            fig.colorbar(sc, ax=ax, label="color")
    plt.tight_layout()
    return fig



