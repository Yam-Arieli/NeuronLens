"""Activation precomputation.

For each neuron in each layer we compute the mean absolute activation over
a (possibly filtered) subset of the dataset.  Results are grouped by
precomputed filter keys so the frontend can switch views without any
re-computation.
"""

from typing import Any, Dict, List, Optional
import hashlib
import json
import numpy as np
import pandas as pd


def _filter_hash(filter_spec: Dict[str, Any]) -> str:
    """Deterministic short hash for a filter specification dict."""
    canonical = json.dumps(filter_spec, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


def _apply_filter(metadata: pd.DataFrame, filter_spec: Dict[str, Any]) -> np.ndarray:
    """Return a boolean mask for rows that satisfy filter_spec.

    filter_spec is a dict like:
        {"column": "label", "op": "eq", "value": "cat"}
        {"column": "age",   "op": "gt", "value": 30}

    Supported ops: eq, ne, lt, le, gt, ge, in, not_in.
    Multiple conditions in the dict are ANDed together — pass a list of dicts
    under key "and" to combine multiple conditions explicitly.
    """
    mask = np.ones(len(metadata), dtype=bool)

    # Support {"and": [cond1, cond2, ...]}
    if "and" in filter_spec:
        for sub in filter_spec["and"]:
            mask &= _apply_filter(metadata, sub)
        return mask

    col = filter_spec["column"]
    op = filter_spec.get("op", "eq")
    value = filter_spec["value"]
    series = metadata[col]

    if op == "eq":
        mask = (series == value).values
    elif op == "ne":
        mask = (series != value).values
    elif op == "lt":
        mask = (series < value).values
    elif op == "le":
        mask = (series <= value).values
    elif op == "gt":
        mask = (series > value).values
    elif op == "ge":
        mask = (series >= value).values
    elif op == "in":
        mask = series.isin(value).values
    elif op == "not_in":
        mask = (~series.isin(value)).values
    else:
        raise ValueError(f"Unknown filter op: {op!r}")

    return mask


def compute_mean_abs_activations(
    layer_activations: List[np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> Dict[int, Any]:
    """Compute per-neuron mean absolute activations for one filter group.

    Args:
        layer_activations: list of arrays (n_samples, n_neurons) per layer.
        mask: boolean array of shape (n_samples,).  If None, use all samples.

    Returns:
        Dict mapping layer_idx -> list of mean-abs values (one per neuron).
    """
    result: Dict[int, Any] = {}
    for l, acts in enumerate(layer_activations):
        if acts is None:
            continue  # e.g. input slot in pre-activation groups
        if mask is not None:
            acts = acts[mask]
        if len(acts) == 0:
            mean_act = np.zeros(acts.shape[1:])
        else:
            # axis=0 reduces over samples; for 4-D (N,C,H,W) this gives (C,H,W)
            mean_act = np.mean(np.abs(acts), axis=0)
        # For conv/image layers (C,H,W): average over channels → (H,W) → flat
        # list of H*W scalars (row-major).  n_display_units = H*W.
        # Linear layers (1-D) are unaffected.
        if mean_act.ndim == 3:
            mean_act = mean_act.mean(axis=0).flatten()  # (C,H,W) → (H*W,)
        result[l] = mean_act.tolist()
    return result


def precompute_activations(
    layer_activations: List[np.ndarray],
    metadata: pd.DataFrame,
    precomputed_filters: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the full activations dict for all filter groups.

    Returns:
        Dict keyed by group label (\"default\" plus a hash per filter).
        Each value is a dict mapping layer_idx -> list of mean-abs values.
    """
    result: Dict[str, Any] = {}

    # Default: all samples
    result["default"] = compute_mean_abs_activations(layer_activations)

    if precomputed_filters:
        for fspec in precomputed_filters:
            key = "filter_" + _filter_hash(fspec)
            mask = _apply_filter(metadata, fspec)
            result[key] = compute_mean_abs_activations(layer_activations, mask)

    # Convert int keys to str for JSON serialisation
    serialisable: Dict[str, Any] = {}
    for group, layer_dict in result.items():
        serialisable[group] = {str(k): v for k, v in layer_dict.items()}

    return serialisable


def filter_key(filter_spec: Dict[str, Any]) -> str:
    """Public helper to obtain the key used for a given filter spec."""
    return "filter_" + _filter_hash(filter_spec)
