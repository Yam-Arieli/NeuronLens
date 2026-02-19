"""End-to-end test for NeuronLens.

Creates a small synthetic dataset and a PyTorch MLP that includes
BatchNorm and GELU layers, then runs the full NeuronLens pipeline
and writes the output to ./test_output/.

Run with:
    python test_run.py
Then open test_output/index.html in a browser to verify the visualisation.
"""

import os
import sys
import json

import torch.nn as nn

# Make the package importable from the repo root without installing it
sys.path.insert(0, os.path.dirname(__file__))
from neuronlens import NeuronLens
from neuronlens.filters import eq, gt, and_
from neuronlens.utils import make_synthetic_data


def make_model(n_features=10):
    """MLP with a mix of BatchNorm, ReLU and GELU blocks."""
    return nn.Sequential(
        nn.Linear(n_features, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 12),
        nn.GELU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def main():
    print("=== NeuronLens end-to-end test ===\n")

    X, metadata = make_synthetic_data(n_samples=200, n_features=10)
    model = make_model(n_features=10)
    model.eval()  # needed for BatchNorm to run in inference mode

    precomputed_filters = [
        eq("label_str", "cat"),
        eq("label_str", "dog"),
        gt("age", 40),
        and_(eq("label_str", "cat"), gt("age", 40)),
    ]

    viz = NeuronLens(
        model=model,
        dataset=X,
        metadata=metadata,
        max_display_units=200,
        n_reorder_passes=5,
        precomputed_filters=precomputed_filters,
        record_pre_activation=True,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    viz.generate(output_dir=output_dir)

    # ── Validate output files exist ──────────────────────────────────────────
    errors = []

    index_html       = os.path.join(output_dir, "index.html")
    network_json     = os.path.join(output_dir, "data", "network.json")
    activations_json = os.path.join(output_dir, "data", "activations.json")

    for path in [index_html, network_json, activations_json]:
        if not os.path.exists(path):
            errors.append(f"MISSING: {path}")

    # ── Validate network.json structure ──────────────────────────────────────
    if os.path.exists(network_json):
        with open(network_json) as f:
            net = json.load(f)
        assert "layers" in net, "network.json missing 'layers'"
        assert "edges"  in net, "network.json missing 'edges'"
        # 5 display layers: input + 4 linear blocks
        assert len(net["layers"]) == 5, (
            f"Expected 5 layers (input + 4 linear blocks), got {len(net['layers'])}"
        )
        assert len(net["edges"]) == 4, (
            f"Expected 4 edge matrices, got {len(net['edges'])}"
        )
        for l in net["layers"]:
            assert "perm"        in l, f"Layer {l['name']} missing 'perm'"
            assert "n_neurons"   in l, f"Layer {l['name']} missing 'n_neurons'"
            assert "block_label" in l, f"Layer {l['name']} missing 'block_label'"
            assert "block_type"  in l, f"Layer {l['name']} missing 'block_type'"

        # Spot-check block labels for the known model architecture
        labels = [l["block_label"] for l in net["layers"]]
        assert labels[0] == "Input",                            f"Layer 0 label: {labels[0]}"
        assert "BatchNorm1d" in labels[1],                      f"Layer 1 label: {labels[1]}"
        assert "GELU"        in labels[2],                      f"Layer 2 label: {labels[2]}"
        assert "ReLU"        in labels[3],                      f"Layer 3 label: {labels[3]}"
        assert labels[4]     == "Linear(8\u21922)",             f"Layer 4 label: {labels[4]}"

        types = [l["block_type"] for l in net["layers"]]
        assert types[0] == "input",                    f"Layer 0 type: {types[0]}"
        assert types[1] == "linear+norm+activation",   f"Layer 1 type: {types[1]}"
        assert types[2] == "linear+activation",        f"Layer 2 type: {types[2]}"
        assert types[3] == "linear+activation",        f"Layer 3 type: {types[3]}"
        assert types[4] == "linear",                   f"Layer 4 type: {types[4]}"

        print(f"  network.json: OK — {len(net['layers'])} layers, block labels verified")

    # ── Validate activations.json structure ──────────────────────────────────
    if os.path.exists(activations_json):
        with open(activations_json) as f:
            act = json.load(f)
        groups = act["groups"]
        assert "default" in groups, "activations.json missing 'default' group"
        assert len(groups) == 1 + len(precomputed_filters), (
            f"Expected {1 + len(precomputed_filters)} groups, got {len(groups)}"
        )
        for gname, gdata in groups.items():
            assert len(gdata) == 5, (
                f"Group {gname}: expected 5 layers (input + 4 blocks), got {len(gdata)}"
            )
        # record_pre_activation=True should produce pre_activation_groups
        assert "pre_activation_groups" in act, (
            "activations.json missing 'pre_activation_groups' (record_pre_activation=True)"
        )
        pre_groups = act["pre_activation_groups"]
        for gname, gdata in pre_groups.items():
            # Only blocks 1-4 have pre-activation data (input has none)
            assert len(gdata) == 4, (
                f"pre_activation group {gname}: expected 4 entries, got {len(gdata)}"
            )
        print(f"  activations.json: OK — {len(groups)} filter groups, pre_activation_groups present")

    # ── Validate index.html is non-trivial ───────────────────────────────────
    if os.path.exists(index_html):
        with open(index_html) as f:
            html = f.read()
        assert "NeuronLens"   in html
        assert "main-canvas"  in html
        assert "block_label"  in html, "block_label not found in inlined HTML"
        print(f"  index.html: OK — {len(html):,} bytes")

    if errors:
        print("\nFAILURES:")
        for e in errors:
            print(" ", e)
        sys.exit(1)
    else:
        print(f"\nAll checks passed.")
        print(f"Open {os.path.abspath(index_html)} in your browser to inspect.")


if __name__ == "__main__":
    main()
