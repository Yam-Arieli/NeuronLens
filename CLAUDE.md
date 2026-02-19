# NeuronLens — Neural Network Activation Visualizer

## Overview

NeuronLens is a Python tool that takes a trained neural network and a labeled dataset, processes activations through the network, reorders neurons within each layer for interpretability, and generates a self-contained interactive HTML visualization.

The prototype currently supports **fully-connected (dense) layers**. The architecture is designed to make it straightforward to extend to other layer types (e.g., CNN, LSTM, Attention) in the future.

---

## What the Tool Does

### Inputs
- A trained model (PyTorch or similar)
- A dataset of observations (tabular)
- Metadata for each observation (e.g., label, age, source — arbitrary key-value columns used for filtering)

### Outputs
- A folder containing:
  - `index.html` — the interactive visualization
  - `data/network.json` — precomputed structure: layer sizes, reordered neuron indices, edge weights
  - `data/activations.json` — precomputed activation statistics per neuron per metadata group/filter

All heavy computation (neuron reordering, activation aggregation) happens in Python at generation time. The HTML/JS side only reads precomputed data and renders it.

---

## Neuron Reordering

### Motivation
In a linear layer, the row order of the weight matrix is arbitrary. We want to reorder neurons within each layer so that neurons that "fire together" are visually adjacent, and so that the overall network has minimal cross-connections between layers.

### Crossing Score
Define a **crossing score** between two edges: edge (i → j) and edge (i' → j') cross if `(i < i') XOR (j < j')` (one goes "over" the other). The crossing score contribution of this pair is `|w_{i,j}| * |w_{i',j'}|` using the absolute values of the corresponding weights.

The total crossing score of the network is the sum of crossing contributions over all crossing pairs, across all adjacent layer pairs.

### Reordering Algorithm
Find permutations of neurons within each layer (simultaneously across all layers) that minimize the total crossing score. Since global optimization is expensive, use a practical iterative approach:
- Initialize with the original order
- For each layer (iterating multiple passes), try local swaps of adjacent neuron pairs and accept if the crossing score decreases
- Repeat until convergence or a fixed number of iterations

The reordered indices are saved to `data/network.json`. The JS visualizer uses these to render neurons in the reordered positions.

**Design note:** The reordering algorithm should be modular so it can be swapped for a better algorithm in the future.

---

## Activation Precomputation

For each neuron in each layer, compute the average absolute activation value over the dataset (or a filtered subset). These averages are stored grouped by metadata combinations so the frontend can switch views instantly without recomputation.

Store activations in `data/activations.json` structured as:
```
{
  "default": { layer_idx: [avg_val_per_neuron] },
  "filter_hash_abc": { layer_idx: [...] },
  ...
}
```

All filters must be declared upfront at generation time via `precomputed_filters`. The JS side only performs lookups into precomputed data — no raw per-observation data is shipped to the browser. This keeps the output lean while still supporting all filtering functionality the user needs.

---

## Large Layer Handling

If a layer has more neurons than a configurable threshold (default: 200), it is rendered in **aggregated mode**: neurons are grouped into fixed-size buckets and the displayed value is the mean activation of each bucket. The bucket size is `ceil(n_neurons / max_display_units)`.

This applies both to the visual rendering and to the hover interaction (see below).

---

## Interactive Visualization (HTML/JS)

### Default View
Render the network as a layered diagram (left to right). Each layer is a vertical column of neurons (or aggregated units). Edge thickness between layers is proportional to weight magnitude. Neuron brightness/glow intensity is proportional to its average activation value over the full dataset.

### Metadata Filtering (Single Filter)
The user can filter the dataset by metadata (e.g., `label == "cat"` or `age > 30`). The visualization updates neuron colors/brightness to reflect activations for the filtered subset only.

### Two-Filter Comparison Mode
The user can define two filter queries simultaneously (e.g., Filter A and Filter B), each assigned a color (e.g., blue and red). For each neuron, the color is interpolated between the two filter colors proportionally to the ratio of activations: if Filter A activation is `a` and Filter B activation is `b`, the hue is `b / (a + b)` toward color B. Brightness is proportional to `a + b` (normalized), so neurons with low activation in both filters appear dim regardless of color.

### Hover on a Neuron
When hovering over neuron `k` in layer `l`, the visualization propagates that neuron's activation forward through the rest of the network:

1. All neurons and edges in layers **before** `l`, and all neurons in layer `l` **except** `k`, are dimmed.
2. The hovered neuron `k` is shown at its normal brightness (from the active filter).
3. Edges from `k` to layer `l+1` are shown at their normal weight-based styling.
4. Neurons in layer `l+1` are **recolored** by their received signal from `k` only:
   `signal[l+1][j] = |w[k, j]| * activation[k]`
5. Edges from layer `l+1` to `l+2` are colored by `|w[i, j]| * signal[l+1][i]` (normalized), showing how strongly each edge carries the propagated signal.
6. Neurons in layer `l+2`, `l+3`, … are colored by the iteratively propagated signal:
   `signal[l+n][m] = Σ_i |w[i, m]| * signal[l+n-1][i]`
7. All downstream edges and neurons are shown at **full opacity** (not dimmed), scaled by their propagated signal magnitude.

The goal is to let the user understand how a single neuron's activation propagates and influences all subsequent layers of the network.

### Rendering Architecture
- Pure HTML + CSS + JS (no framework required, but D3.js or similar is acceptable)
- All data loaded from `data/*.json` at page load
- No server required — fully static

---

## Python Module Interface

```python
from neuronlens import NeuronLens

viz = NeuronLens(model, dataset, metadata)
viz.generate(output_dir="./neuronlens_output")
```

- `model`: a callable that returns intermediate activations per layer (adapter pattern — see extensibility)
- `dataset`: array-like of shape `(n_samples, n_features)`
- `metadata`: `pd.DataFrame` of shape `(n_samples, n_meta_columns)`

Configuration options (passed to constructor or `generate()`):
- `max_display_units` (int, default 200): threshold for aggregated layer rendering
- `n_reorder_passes` (int, default 10): iterations for the neuron reordering algorithm
- `precomputed_filters` (list of dicts): metadata filter conditions to precompute activations for

---

## Extensibility Notes

The following features are **not** in scope for the prototype but the code must be structured to support them cleanly:

1. **More layer types**: The model adapter should return a list of `(layer_name, layer_type, activations, weights)` tuples. New layer types (Conv, Attention, etc.) should only require adding a new adapter and a new renderer — not changing core logic.

2. **Multiple epochs**: The data format in `network.json` and `activations.json` should include an optional `epoch` dimension so that epoch-indexed data can be added later and the frontend can add a timeline slider without restructuring the data files.

---

## File Structure

```
neuronlens/
├── __init__.py
├── core.py            # NeuronLens main class
├── reorder.py         # Neuron reordering algorithm
├── activations.py     # Activation extraction and aggregation
├── export.py          # JSON data file generation
├── render/
│   ├── template.html  # Base HTML template
│   ├── main.js        # Visualization logic
│   └── style.css      # Styles
└── adapters/
    ├── base.py        # Abstract model adapter
    └── pytorch.py     # PyTorch adapter
```