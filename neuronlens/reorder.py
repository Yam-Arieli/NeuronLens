"""Neuron reordering via crossing-score minimization.

The crossing score between edge (i->j) and edge (i'->j') is positive when
  (i < i') XOR (j < j')
i.e. one edge crosses the other.  Its contribution is |w_ij| * |w_i'j'|.

We minimise the total crossing score with an iterative local-swap heuristic:
for each layer in turn we try swapping every pair of adjacent neurons and
keep the swap if it strictly reduces the global crossing score.  We repeat
for `n_passes` full sweeps until convergence.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def _crossing_score_pair(W: np.ndarray, perm_in: np.ndarray, perm_out: np.ndarray) -> float:
    """Compute the crossing score for one pair of adjacent layers.

    W has shape (n_in, n_out).  perm_in and perm_out give the current neuron
    orderings for the input and output layers respectively.
    """
    W_reordered = W[np.ix_(perm_in, perm_out)]
    n_in, n_out = W_reordered.shape
    score = 0.0
    absW = np.abs(W_reordered)
    # Vectorised: for each edge pair (i,j) vs (i',j') where i<i', count crossings
    for i in range(n_in):
        for j in range(n_out):
            if absW[i, j] == 0:
                continue
            # Edges from rows > i that cross: row i' > i, col j' < j
            score += absW[i, j] * absW[i + 1:, :j].sum()
    return score


def _delta_swap_in(W: np.ndarray, perm: np.ndarray, k: int) -> float:
    """Change in crossing score for one layer when swapping neurons k and k+1
    in the *input* (row) dimension of weight matrix W.

    W is already in original (un-permuted) indexing; perm gives current order.
    Returns delta = new_score - old_score  (negative means improvement).
    """
    absW = np.abs(W)
    row_k = absW[perm[k]]
    row_k1 = absW[perm[k + 1]]
    # Edges from k cross edges from k+1 when col(k) > col(k+1)
    # (i.e. the edge from the lower row goes to a higher column than the edge
    # from the higher row).  Swapping k <-> k+1 inverts those crossings.
    # Delta = 2 * (crossings_removed - crossings_added)
    # crossings between rows k and k+1 that currently exist (row k lower):
    #   for each column pair (j, j') with j < j':
    #     current crossing: row_k[j'] * row_k1[j]
    #     after swap crossing: row_k[j] * row_k1[j']
    # Total delta = 2 * sum_{j<j'} (row_k[j]*row_k1[j'] - row_k[j']*row_k1[j])
    delta = 0.0
    n_out = len(row_k)
    for j in range(n_out):
        for jp in range(j + 1, n_out):
            delta += 2.0 * (row_k[j] * row_k1[jp] - row_k[jp] * row_k1[j])
    return delta


def _delta_swap_out(W: np.ndarray, perm: np.ndarray, k: int) -> float:
    """Change in crossing score when swapping neurons k and k+1 in the
    *output* (col) dimension of weight matrix W."""
    absW = np.abs(W)
    col_k = absW[:, perm[k]]
    col_k1 = absW[:, perm[k + 1]]
    delta = 0.0
    n_in = len(col_k)
    for i in range(n_in):
        for ip in range(i + 1, n_in):
            delta += 2.0 * (col_k[i] * col_k1[ip] - col_k[ip] * col_k1[i])
    return delta


def reorder_neurons(
    weights: List[np.ndarray],
    n_passes: int = 10,
) -> List[np.ndarray]:
    """Find neuron permutations that minimise the total crossing score.

    Args:
        weights: List of weight matrices W_l of shape (n_in_l, n_out_l).
                 len(weights) == n_layers - 1.
        n_passes: Number of full sweeps over all layers.

    Returns:
        List of permutation arrays, one per layer (len == n_layers).
        perms[l][k] gives the original neuron index that appears at position k.
    """
    if not weights:
        return []

    # Determine layer sizes from weight dimensions
    n_layers = len(weights) + 1
    layer_sizes = [weights[0].shape[0]] + [w.shape[1] for w in weights]

    perms = [np.arange(s) for s in layer_sizes]

    for _ in range(n_passes):
        changed = False
        for l in range(n_layers):
            size = layer_sizes[l]
            for k in range(size - 1):
                delta = 0.0
                # Contribution from the left weight matrix (l-1 -> l)
                if l > 0:
                    delta += _delta_swap_out(weights[l - 1], perms[l], k)
                # Contribution from the right weight matrix (l -> l+1)
                if l < n_layers - 1:
                    delta += _delta_swap_in(weights[l], perms[l], k)
                if delta < 0:
                    perms[l][k], perms[l][k + 1] = perms[l][k + 1], perms[l][k]
                    changed = True
        if not changed:
            break

    return perms


def reorder_neurons_fast(
    weights: List[np.ndarray],
    n_passes: int = 10,
) -> List[np.ndarray]:
    """Vectorised version of reorder_neurons for larger networks.

    Uses numpy broadcasting to compute swap deltas in bulk instead of
    inner Python loops, making it substantially faster for large layers.

    Entries in ``weights`` may be ``None`` to indicate that the connection
    between those two layers should be skipped (e.g. conv-involving edges).
    Layers adjacent only to ``None`` weights get a trivial identity
    permutation of size 1 (overridden downstream in export.py for spatial
    layers).
    """
    if not weights:
        return []

    n_layers = len(weights) + 1

    # Determine layer sizes from non-None weights only.
    layer_sizes: List[int] = [None] * n_layers  # type: ignore[list-item]
    for i, w in enumerate(weights):
        if w is None:
            continue
        if layer_sizes[i] is None:
            layer_sizes[i] = w.shape[0]
        if layer_sizes[i + 1] is None:
            layer_sizes[i + 1] = w.shape[1]
    # Fall back to 1 for layers only adjacent to None weights (conv layers).
    layer_sizes = [s if s is not None else 1 for s in layer_sizes]

    perms = [np.arange(s) for s in layer_sizes]
    abs_weights = [np.abs(w) if w is not None else None for w in weights]

    def delta_swap_out_vec(absW, perm, k):
        col_k  = absW[:, perm[k]]
        col_k1 = absW[:, perm[k + 1]]
        outer  = np.outer(col_k, col_k1)
        return 2.0 * (np.tril(outer, -1).sum() - np.triu(outer, 1).sum())

    def delta_swap_in_vec(absW, perm, k):
        row_k  = absW[perm[k]]
        row_k1 = absW[perm[k + 1]]
        outer  = np.outer(row_k, row_k1)
        return 2.0 * (np.tril(outer, -1).sum() - np.triu(outer, 1).sum())

    for _ in range(n_passes):
        changed = False
        for l in range(n_layers):
            size = layer_sizes[l]
            for k in range(size - 1):
                delta = 0.0
                if l > 0 and abs_weights[l - 1] is not None:
                    delta += delta_swap_out_vec(abs_weights[l - 1], perms[l], k)
                if l < n_layers - 1 and abs_weights[l] is not None:
                    delta += delta_swap_in_vec(abs_weights[l], perms[l], k)
                if delta < 0:
                    perms[l][k], perms[l][k + 1] = perms[l][k + 1], perms[l][k]
                    changed = True
        if not changed:
            break

    return perms


def reorder_neurons_by_class(
    layer_activations: List[np.ndarray],
    metadata: pd.DataFrame,
    column: str,
    layer_info: List[Dict],
) -> List[np.ndarray]:
    """Return permutations that sort neurons by class preference (center-of-mass).

    For each linear layer, computes the per-class mean |activation| for every
    neuron, then assigns a 1-D score via center-of-mass over the sorted class
    indices.  Neurons are sorted ascending so that neurons most correlated with
    class 0 (alphabetically first) appear at the top and those correlated with
    the last class appear at the bottom.

    Spatial (conv/input) layers always receive an identity permutation.

    Args:
        layer_activations: List of activation arrays, one per displayed layer.
            Shape is ``(n_samples, n_units)`` for linear/input layers.
        metadata: DataFrame with one row per sample.  Must contain *column*.
        column: The metadata column whose unique values define the classes.
        layer_info: List of layer info dicts (same length as layer_activations).
            Each dict has at minimum ``"type"`` and optionally ``"spatial_h"``.

    Returns:
        List of permutation arrays, one per layer.  ``perms[l][k]`` is the
        original neuron index that should be shown at display position *k*.
    """
    classes = sorted(metadata[column].dropna().unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    eps = 1e-8

    def _is_spatial(info: Dict) -> bool:
        return info.get("type") in ("conv", "input") and "spatial_h" in info

    perms: List[np.ndarray] = []
    for acts, info in zip(layer_activations, layer_info):
        n_units = acts.shape[1] if acts.ndim == 2 else acts.shape[0]

        if _is_spatial(info):
            perms.append(np.arange(n_units))
            continue

        # Compute per-class mean |activation| — shape (n_units,)
        abs_acts = np.abs(acts)  # (n_samples, n_units)
        class_means = np.zeros((len(classes), n_units), dtype=np.float64)
        for c, idx in class_to_idx.items():
            mask = (metadata[column] == c).values
            if mask.any():
                class_means[idx] = abs_acts[mask].mean(axis=0)

        # Center-of-mass score: Σ_c (c_index * mean_c[j]) / (Σ_c mean_c[j] + ε)
        c_indices = np.arange(len(classes), dtype=np.float64).reshape(-1, 1)
        total = class_means.sum(axis=0)  # (n_units,)
        score = (c_indices * class_means).sum(axis=0) / (total + eps)

        perms.append(np.argsort(score, kind="stable"))

    return perms
