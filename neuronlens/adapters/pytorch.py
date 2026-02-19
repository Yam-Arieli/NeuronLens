from typing import List, Tuple
import numpy as np
from .base import ModelAdapter


class PyTorchAdapter(ModelAdapter):
    """Adapter for PyTorch nn.Sequential or any module with named linear layers.

    Supported layer types: nn.Linear.  The adapter walks child modules in
    order and collects every nn.Linear it finds.  Hooks capture activations
    (post-activation) during a forward pass.
    """

    def __init__(self, model):
        import torch
        import torch.nn as nn

        self._torch = torch
        self._nn = nn
        self._model = model
        self._model.eval()

        # Discover linear layers in order
        self._linear_layers: List = []
        self._layer_names: List[str] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._linear_layers.append(module)
                self._layer_names.append(name or f"layer_{len(self._layer_names)}")

        if not self._linear_layers:
            raise ValueError("No nn.Linear layers found in the model.")

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def get_layers(self) -> List[Tuple[str, str]]:
        return [(name, "linear") for name in self._layer_names]

    def get_activations(self, inputs: np.ndarray) -> List[np.ndarray]:
        torch = self._torch
        activations: List[np.ndarray] = []
        hooks = []

        def make_hook(storage):
            def hook(module, inp, output):
                storage.append(output.detach().cpu().numpy())
            return hook

        storage_per_layer = [[] for _ in self._linear_layers]
        for layer, storage in zip(self._linear_layers, storage_per_layer):
            hooks.append(layer.register_forward_hook(make_hook(storage)))

        x = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            self._model(x)

        for hook in hooks:
            hook.remove()

        for storage in storage_per_layer:
            # storage contains one array per batch; concatenate along axis 0
            activations.append(np.concatenate(storage, axis=0))

        return activations

    def get_weights(self) -> List[np.ndarray]:
        """Return weight matrices W of shape (n_in, n_out) for each layer transition.

        PyTorch stores weights as (out_features, in_features), so we transpose.
        """
        weights = []
        for layer in self._linear_layers:
            w = layer.weight.detach().cpu().numpy()  # (out, in)
            weights.append(w.T)  # (in, out)
        return weights
