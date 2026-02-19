"""JSON export helpers.

Generates data/network.json and data/activations.json from precomputed data.
"""

import json
import os
from typing import Any, Dict, List, Optional
import numpy as np


def _to_json_safe(obj):
    """Recursively convert numpy scalars/arrays to Python natives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


def build_network_json(
    layer_info: List[Dict[str, Any]],  # list of dicts from adapter.get_layers()
    layer_sizes: List[int],
    perms: List[np.ndarray],
    weights: List[np.ndarray],          # (n_in, n_out) original indexing
    max_display_units: int = 200,
) -> Dict[str, Any]:
    """Build the network.json data structure.

    Args:
        layer_info: list of info dicts, one per layer (including synthetic
            input entry).  Each dict must have ``name``, ``type``,
            ``block_label``, and ``block_type``.
        layer_sizes: number of neurons per layer.
        perms: permutation arrays (display_pos -> original_idx) per layer.
        weights: weight matrices between adjacent layers, original indexing.
        max_display_units: threshold for aggregated rendering.

    Returns:
        dict ready for JSON serialisation.
    """
    layers_out = []
    for l, info in enumerate(layer_info):
        name        = info["name"]
        ltype       = info.get("type", "linear")
        block_label = info.get("block_label", name)
        block_type  = info.get("block_type", "linear")

        n = layer_sizes[l]
        aggregated  = n > max_display_units
        bucket_size = int(np.ceil(n / max_display_units)) if aggregated else 1
        n_display   = int(np.ceil(n / bucket_size))
        layers_out.append({
            "name":           name,
            "type":           ltype,
            "block_label":    block_label,
            "block_type":     block_type,
            "n_neurons":      n,
            "aggregated":     aggregated,
            "bucket_size":    bucket_size,
            "n_display_units": n_display,
            "perm":           perms[l].tolist(),  # perm[display_pos] = original_idx
        })

    # Edges: store as weight matrix for each layer transition, in display order.
    # edges[l][display_i][display_j] = weight from display unit i (layer l)
    #                                  to display unit j (layer l+1)
    edges_out = []
    for l, W in enumerate(weights):
        perm_in  = perms[l]
        perm_out = perms[l + 1]
        W_reordered = W[np.ix_(perm_in, perm_out)]  # (n_in, n_out)
        edges_out.append(W_reordered.tolist())

    return {
        "layers": layers_out,
        "edges":  edges_out,
        "max_display_units": max_display_units,
    }


def build_activations_json(
    activations_by_group: Dict[str, Any],
    filter_metadata: Optional[List[Dict[str, Any]]] = None,
    pre_activations_by_group: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the activations.json data structure.

    Args:
        activations_by_group: post-activation stats keyed by group name.
        filter_metadata: list of filter specs (for building filter_index).
        pre_activations_by_group: optional post-linear (pre-norm/activation)
            stats, present only when record_pre_activation=True.
    """
    from .activations import filter_key

    filter_index = {}
    if filter_metadata:
        for fspec in filter_metadata:
            k = filter_key(fspec)
            filter_index[k] = fspec

    result: Dict[str, Any] = {
        "groups":       activations_by_group,
        "filter_index": filter_index,
    }
    if pre_activations_by_group is not None:
        result["pre_activation_groups"] = pre_activations_by_group

    return result


def write_output(
    output_dir: str,
    network_data: Dict[str, Any],
    activations_data: Dict[str, Any],
    html_content: str,
) -> None:
    """Write all output files to disk."""
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "network.json"), "w") as f:
        json.dump(_to_json_safe(network_data), f)

    with open(os.path.join(data_dir, "activations.json"), "w") as f:
        json.dump(_to_json_safe(activations_data), f)

    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html_content)
