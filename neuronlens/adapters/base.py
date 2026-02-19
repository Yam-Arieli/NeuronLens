from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class ModelAdapter(ABC):
    """Abstract base class for model adapters.

    Adapters wrap a trained model and expose a uniform interface for
    extracting per-layer activations and weights.
    """

    @abstractmethod
    def get_layers(self) -> List[Tuple[str, str]]:
        """Return a list of (layer_name, layer_type) for each supported layer."""

    @abstractmethod
    def get_activations(self, inputs: np.ndarray) -> List[np.ndarray]:
        """Run inputs through the model and return activations per layer.

        Returns:
            List of arrays, one per layer, each of shape (n_samples, n_neurons).
        """

    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        """Return weight matrices between adjacent layers.

        Returns:
            List of 2-D arrays W where W[i, j] is the weight from neuron i in
            layer l to neuron j in layer l+1. Length is len(layers) - 1.
        """
