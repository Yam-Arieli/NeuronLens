from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np


class ModelAdapter(ABC):
    """Abstract base class for model adapters.

    Adapters wrap a trained model and expose a uniform interface for
    extracting per-layer activations and weights.
    """

    @abstractmethod
    def get_layers(self) -> List[Dict[str, Any]]:
        """Return one info dict per display block/layer.

        Each dict must contain at minimum:

        * ``name`` (str) — short identifier, e.g. ``"0"`` or ``"fc1"``
        * ``type`` (str) — layer type, e.g. ``"linear"``
        * ``block_label`` (str) — human-readable label shown in the UI,
          e.g. ``"Linear(32→16) · BatchNorm · ReLU"``
        * ``block_type`` (str) — structural tag, one of:
          ``"linear"``, ``"linear+norm"``, ``"linear+activation"``,
          ``"linear+norm+activation"``
        """

    @abstractmethod
    def get_activations(self, inputs: np.ndarray) -> List[np.ndarray]:
        """Run inputs through the model and return activations per layer/block.

        Returns:
            List of arrays, one per block, each of shape
            ``(n_samples, n_neurons)``.  Values should be post-activation
            (i.e. the actual output that flows to the next layer).
        """

    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        """Return weight matrices between adjacent layers.

        Returns:
            List of 2-D arrays W where ``W[i, j]`` is the weight from
            neuron ``i`` in layer ``l`` to neuron ``j`` in layer ``l+1``.
            Length equals ``len(get_layers())``, one matrix per block
            (the linear layer's weights, shape ``(n_in, n_out)``).
        """
