"""General-purpose utilities for NeuronLens.

Useful for quick experiments, demos, and smoke-tests.
"""

import numpy as np
import pandas as pd


def make_synthetic_data(
    n_samples: int = 200,
    n_features: int = 10,
    *,
    seed: int = 42,
) -> tuple:
    """Generate a small synthetic tabular dataset for testing NeuronLens.

    Produces a feature matrix ``X`` and a ``metadata`` DataFrame with three
    columns:

    * ``label`` (int 0/1) — binary class based on the sign of the first two
      features
    * ``label_str`` (str "cat"/"dog") — string version of the label
    * ``age`` (int 20–59) — random integer covariate

    Args:
        n_samples: Number of rows.
        n_features: Number of input features.
        seed: Random seed for reproducibility.

    Returns:
        ``(X, metadata)`` where ``X`` is a ``float32`` numpy array of shape
        ``(n_samples, n_features)`` and ``metadata`` is a ``pd.DataFrame``.

    Example::

        from neuronlens.utils import make_synthetic_data
        X, metadata = make_synthetic_data(n_samples=500, n_features=16)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    labels = (X[:, 0] + X[:, 1] > 0).astype(int)
    ages = rng.integers(20, 60, size=n_samples)
    metadata = pd.DataFrame({
        "label": labels,
        "label_str": ["cat" if l == 1 else "dog" for l in labels],
        "age": ages,
    })
    return X, metadata
