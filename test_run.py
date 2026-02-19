"""End-to-end test for NeuronLens.

Creates a small synthetic dataset and a simple PyTorch MLP, then runs the
full NeuronLens pipeline and writes the output to ./test_output/.

Run with:
    python test_run.py
Then open test_output/index.html in a browser to verify the visualisation.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Make the package importable from the repo root without installing it
sys.path.insert(0, os.path.dirname(__file__))
from neuronlens import NeuronLens


def make_synthetic_data(n_samples=200, n_features=10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    # Two classes: 0 and 1
    labels = (X[:, 0] + X[:, 1] > 0).astype(int)
    ages = rng.integers(20, 60, size=n_samples)
    metadata = pd.DataFrame({
        "label": labels,
        "label_str": ["cat" if l == 1 else "dog" for l in labels],
        "age": ages,
    })
    return X, metadata


def make_model(n_features=10):
    model = nn.Sequential(
        nn.Linear(n_features, 16),
        nn.ReLU(),
        nn.Linear(16, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    # Use random weights (untrained) — that's fine for visualisation testing
    return model


def main():
    print("=== NeuronLens end-to-end test ===\n")

    X, metadata = make_synthetic_data()
    model = make_model()

    precomputed_filters = [
        {"column": "label_str", "op": "eq", "value": "cat"},
        {"column": "label_str", "op": "eq", "value": "dog"},
        {"column": "age", "op": "gt", "value": 40},
    ]

    viz = NeuronLens(
        model=model,
        dataset=X,
        metadata=metadata,
        max_display_units=200,
        n_reorder_passes=5,
        precomputed_filters=precomputed_filters,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    viz.generate(output_dir=output_dir)

    # ── Validate output files exist ──────────────────────────────────────────
    errors = []

    index_html = os.path.join(output_dir, "index.html")
    network_json = os.path.join(output_dir, "data", "network.json")
    activations_json = os.path.join(output_dir, "data", "activations.json")

    for path in [index_html, network_json, activations_json]:
        if not os.path.exists(path):
            errors.append(f"MISSING: {path}")

    # ── Validate network.json structure ─────────────────────────────────────
    if os.path.exists(network_json):
        with open(network_json) as f:
            net = json.load(f)
        assert "layers" in net, "network.json missing 'layers'"
        assert "edges" in net, "network.json missing 'edges'"
        assert len(net["layers"]) == 5, f"Expected 5 layers (input + 4 linear), got {len(net['layers'])}"
        assert len(net["edges"]) == 4, f"Expected 4 edge matrices, got {len(net['edges'])}"
        for l in net["layers"]:
            assert "perm" in l, f"Layer {l['name']} missing 'perm'"
            assert "n_neurons" in l
        print(f"  network.json: OK — {len(net['layers'])} layers")

    # ── Validate activations.json structure ──────────────────────────────────
    if os.path.exists(activations_json):
        with open(activations_json) as f:
            act = json.load(f)
        groups = act["groups"]
        assert "default" in groups, "activations.json missing 'default' group"
        # Each filter should produce one entry
        assert len(groups) == 1 + len(precomputed_filters), (
            f"Expected {1 + len(precomputed_filters)} groups, got {len(groups)}"
        )
        for gname, gdata in groups.items():
            assert len(gdata) == 5, (
                f"Group {gname}: expected 5 layers (input + 4 linear), got {len(gdata)}"
            )
        print(f"  activations.json: OK — {len(groups)} filter groups")

    # ── Validate index.html is non-trivial ───────────────────────────────────
    if os.path.exists(index_html):
        with open(index_html) as f:
            html = f.read()
        assert "NeuronLens" in html
        assert "main-canvas" in html
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
