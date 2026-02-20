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


def _is_matrix_layer(info: Dict[str, Any]) -> bool:
    """True for layers rendered as 2D spatial matrices (conv/input with spatial dims)."""
    return info.get("type") in ("conv", "input") and "spatial_h" in info


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

        if _is_matrix_layer(info):
            # Conv/image layers: rendered as a 2D grid of H*W cells.
            # n_display_units = H*W (one display unit per spatial cell).
            # Spatial positions are never reordered.
            n_cells     = info["spatial_h"] * info["spatial_w"]
            aggregated  = False
            bucket_size = 1
            n_display   = n_cells
            entry_perm  = list(range(n_cells))
        else:
            aggregated  = n > max_display_units
            bucket_size = int(np.ceil(n / max_display_units)) if aggregated else 1
            n_display   = int(np.ceil(n / bucket_size))
            entry_perm  = perms[l].tolist()

        entry = {
            "name":            name,
            "type":            ltype,
            "block_label":     block_label,
            "block_type":      block_type,
            "n_neurons":       n,
            "aggregated":      aggregated,
            "bucket_size":     bucket_size,
            "n_display_units": n_display,
            "perm":            entry_perm,
        }
        # Conv/image layers include spatial dimensions for heatmap rendering
        if "channels" in info:
            entry["channels"]  = info["channels"]
        if "spatial_h" in info:
            entry["spatial_h"] = info["spatial_h"]
        if "spatial_w" in info:
            entry["spatial_w"] = info["spatial_w"]
        layers_out.append(entry)

    # Edges: weight matrix for each layer transition, in display order.
    # edges[l][display_i][display_j] = weight from display unit i (layer l)
    #                                  to display unit j (layer l+1).
    #
    # No lines are ever drawn to/from conv layers, but we STORE Conv→Linear
    # weights so that hover signal can propagate from conv cells to linear
    # neurons.  Conv→Conv and Input→Conv are set to null (no propagation).
    edges_out = []
    for l, W in enumerate(weights):
        from_matrix = _is_matrix_layer(layer_info[l])
        to_matrix   = _is_matrix_layer(layer_info[l + 1])

        if from_matrix and to_matrix:
            # Conv→Conv / Input→Conv: no edges, no propagation
            edges_out.append(None)
        elif from_matrix and not to_matrix:
            # Conv→Linear: store per-cell weights for hover propagation.
            # perm_in is identity (spatial positions never reordered).
            # Guard: if W.shape[0] ≠ n_cells (e.g. MaxPool between conv hook and
            # linear input changes spatial dims), skip edge storage rather than crash.
            n_cells  = layer_info[l]["spatial_h"] * layer_info[l]["spatial_w"]
            if W.shape[0] != n_cells:
                edges_out.append(None)
            else:
                perm_in  = np.arange(n_cells)
                perm_out = perms[l + 1]
                W_reordered = W[np.ix_(perm_in, perm_out)]
                edges_out.append(W_reordered.tolist())
        elif not from_matrix and to_matrix:
            # Linear→Conv (unusual): no edges
            edges_out.append(None)
        else:
            # Linear→Linear: store as before
            perm_in     = perms[l]
            perm_out    = perms[l + 1]
            W_reordered = W[np.ix_(perm_in, perm_out)]
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
