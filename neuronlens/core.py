"""NeuronLens main class."""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .adapters.base import ModelAdapter
from .adapters.pytorch import PyTorchAdapter
from .activations import precompute_activations
from .export import build_network_json, build_activations_json, write_output
from .reorder import reorder_neurons_fast


class NeuronLens:
    """Neural network activation visualizer.

    Args:
        model: A trained model.  Can be a PyTorch nn.Module (auto-detected) or
               a :class:`ModelAdapter` instance for custom wrapping.
        dataset: Array-like of shape (n_samples, n_features).
        metadata: pd.DataFrame of shape (n_samples, n_meta_columns).
        max_display_units: Threshold above which a layer is rendered in
            aggregated (bucketed) mode.  Default 200.
        n_reorder_passes: Iterations for the neuron reordering algorithm.
            Default 10.
        precomputed_filters: List of filter spec dicts to precompute
            activation subsets for.  See activations.py for the spec format.
    """

    def __init__(
        self,
        model,
        dataset,
        metadata: pd.DataFrame,
        *,
        max_display_units: int = 200,
        n_reorder_passes: int = 10,
        precomputed_filters: Optional[List[Dict[str, Any]]] = None,
    ):
        self.metadata = metadata
        self.max_display_units = max_display_units
        self.n_reorder_passes = n_reorder_passes
        self.precomputed_filters = precomputed_filters or []

        # Resolve adapter
        if isinstance(model, ModelAdapter):
            self.adapter = model
        else:
            # Try PyTorch
            try:
                self.adapter = PyTorchAdapter(model)
            except Exception as exc:
                raise TypeError(
                    "model must be a ModelAdapter or a PyTorch nn.Module "
                    f"with at least one nn.Linear layer. Error: {exc}"
                ) from exc

        self.dataset = np.asarray(dataset, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self, output_dir: str = "./neuronlens_output") -> str:
        """Generate output and immediately open index.html in the default browser.

        Equivalent to calling :meth:`generate` followed by opening the file.

        Args:
            output_dir: Directory where index.html and data/ will be written.

        Returns:
            Absolute path to the output directory.
        """
        import webbrowser
        output_dir = self.generate(output_dir)
        webbrowser.open(f"file://{os.path.join(output_dir, 'index.html')}")
        return output_dir

    def generate(self, output_dir: str = "./neuronlens_output") -> str:
        """Run the full pipeline and write output files.

        Args:
            output_dir: Directory where index.html and data/ will be written.

        Returns:
            Absolute path to the output directory.
        """
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[NeuronLens] Extracting activations from {len(self.dataset)} samples…")
        hidden_activations = self.adapter.get_activations(self.dataset)

        print("[NeuronLens] Fetching weights…")
        weights = self.adapter.get_weights()

        # Prepend the input features as the first displayed layer so that
        # perms[l] and edges[l] align correctly (reorder_neurons_fast returns
        # one permutation per "node" in the graph, including the input).
        layer_activations = [self.dataset] + hidden_activations
        layer_info = [("input", "input")] + self.adapter.get_layers()
        layer_sizes = [a.shape[1] for a in layer_activations]

        print(f"[NeuronLens] Reordering neurons ({self.n_reorder_passes} passes)…")
        perms = reorder_neurons_fast(weights, n_passes=self.n_reorder_passes)

        print("[NeuronLens] Precomputing activation statistics…")
        activations_by_group = precompute_activations(
            layer_activations, self.metadata, self.precomputed_filters
        )

        print("[NeuronLens] Building network data…")
        network_data = build_network_json(
            layer_info,
            layer_sizes,
            perms,
            weights,
            max_display_units=self.max_display_units,
        )
        activations_data = build_activations_json(
            activations_by_group, self.precomputed_filters
        )

        print("[NeuronLens] Rendering HTML…")
        html = self._render_html(network_data, activations_data)

        print(f"[NeuronLens] Writing output to {output_dir}…")
        write_output(output_dir, network_data, activations_data, html)

        print(f"[NeuronLens] Done.  Open {os.path.join(output_dir, 'index.html')}")
        return output_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_html(self, network_data: Dict[str, Any], activations_data: Dict[str, Any]) -> str:
        """Load the HTML template and inline JS/CSS/data so the file works via file://."""
        import json as _json
        render_dir = os.path.join(os.path.dirname(__file__), "render")

        with open(os.path.join(render_dir, "template.html")) as f:
            template = f.read()
        with open(os.path.join(render_dir, "main.js")) as f:
            js = f.read()
        with open(os.path.join(render_dir, "style.css")) as f:
            css = f.read()

        html = template.replace("/* {{STYLE}} */", css)
        html = html.replace("// {{NETWORK_DATA}}", _json.dumps(network_data, separators=(",", ":")))
        html = html.replace("// {{ACTIVATIONS_DATA}}", _json.dumps(activations_data, separators=(",", ":")))
        html = html.replace("// {{SCRIPT}}", js)
        return html
