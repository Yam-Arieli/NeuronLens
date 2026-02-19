"""PyTorch model adapter with block-level detection.

Groups sequences of Linear → [Norm] → [Activation] into display blocks.
Activations captured are post-block (post-activation) by default.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from .base import ModelAdapter


# ── Layer-type classification ─────────────────────────────────────────────────

def _norm_types():
    import torch.nn as nn
    return (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d)


def _act_types():
    import torch.nn as nn
    types = [nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.ELU]
    for name in ("SiLU", "Mish", "Hardswish", "Hardtanh", "Softmax"):
        if hasattr(nn, name):
            types.append(getattr(nn, name))
    return tuple(types)


def _container_types():
    import torch.nn as nn
    return (nn.Sequential, nn.ModuleList, nn.ModuleDict)


# ── Block detection ───────────────────────────────────────────────────────────

def _detect_blocks(model, nn) -> List[dict]:
    """Walk model.named_modules() and group into Linear-headed blocks.

    A block is: nn.Linear → [zero or more Norms] → [zero or more Activations].
    The canonical order expected by the spec is Linear → Norm → Activation.
    Norms that appear after an activation are not added to the block.
    Any module not fitting this pattern is skipped without closing the block.
    """
    norm_t = _norm_types()
    act_t  = _act_types()
    ctr_t  = _container_types()

    blocks: List[dict] = []
    current: Optional[dict] = None

    for name, mod in model.named_modules():
        if mod is model or isinstance(mod, ctr_t):
            continue

        if isinstance(mod, nn.Linear):
            if current is not None:
                blocks.append(current)
            current = {
                "linear":      mod,
                "linear_name": name,
                "norms":       [],
                "activations": [],
            }
        elif isinstance(mod, norm_t) and current is not None and not current["activations"]:
            # Norms are grouped only before any activation (canonical order)
            current["norms"].append(mod)
        elif isinstance(mod, act_t) and current is not None:
            current["activations"].append(mod)
        # Standalone modules (Dropout, Flatten, etc.) are skipped without
        # closing the current block, so they attach to the surrounding blocks
        # naturally through the execution graph.

    if current is not None:
        blocks.append(current)

    return blocks


def _block_label(block: dict) -> str:
    lin = block["linear"]
    parts = [f"Linear({lin.in_features}\u2192{lin.out_features})"]
    for m in block["norms"]:
        parts.append(type(m).__name__)
    for m in block["activations"]:
        parts.append(type(m).__name__)
    return " \u00b7 ".join(parts)  # " · ".join(parts)


def _block_type(block: dict) -> str:
    has_norm = bool(block["norms"])
    has_act  = bool(block["activations"])
    if has_norm and has_act:
        return "linear+norm+activation"
    if has_norm:
        return "linear+norm"
    if has_act:
        return "linear+activation"
    return "linear"


def _last_module(block: dict):
    """The last module in execution order (used for post-activation hooking)."""
    if block["activations"]:
        return block["activations"][-1]
    if block["norms"]:
        return block["norms"][-1]
    return block["linear"]


def _make_hook(storage: list):
    def hook(module, inp, output):
        storage.append(output.detach().cpu().numpy())
    return hook


# ── Adapter ───────────────────────────────────────────────────────────────────

class PyTorchAdapter(ModelAdapter):
    """Adapter for PyTorch models with block-level detection.

    Groups ``Linear → [Norm] → [Activation]`` sequences into blocks.
    Activations captured are post-block (post-activation) by default.

    Args:
        model: An ``nn.Module`` (typically ``nn.Sequential`` or any module
               whose children can be walked with ``named_modules()``).
        record_pre_activation: If ``True``, also captures post-linear
            (pre-norm/activation) outputs so dead-neuron analysis is
            possible later via :meth:`get_pre_activations`.
    """

    def __init__(self, model, *, record_pre_activation: bool = False):
        import torch
        import torch.nn as nn

        self._torch = torch
        self._nn = nn
        self._model = model
        self._model.eval()
        self._record_pre = record_pre_activation
        self._pre_activations: Optional[List[np.ndarray]] = None

        self._blocks = _detect_blocks(model, nn)
        if not self._blocks:
            raise ValueError("No nn.Linear layers found in the model.")

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def get_layers(self) -> List[Dict[str, Any]]:
        """Return one info dict per block.

        Keys: ``name``, ``type``, ``block_label``, ``block_type``.
        """
        return [
            {
                "name":        block["linear_name"],
                "type":        "linear",
                "block_label": _block_label(block),
                "block_type":  _block_type(block),
            }
            for block in self._blocks
        ]

    def get_activations(self, inputs: np.ndarray) -> List[np.ndarray]:
        """Run inputs and return post-block (post-activation) outputs per block."""
        torch = self._torch
        n = len(self._blocks)

        post_stores = [[] for _ in range(n)]
        pre_stores  = [[] for _ in range(n)] if self._record_pre else None

        hooks = []
        for i, block in enumerate(self._blocks):
            # Post-activation: hook on the last module of the block
            hooks.append(_last_module(block).register_forward_hook(
                _make_hook(post_stores[i])
            ))
            # Pre-activation: hook on the linear layer (post-linear)
            if self._record_pre:
                hooks.append(block["linear"].register_forward_hook(
                    _make_hook(pre_stores[i])
                ))

        x = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            self._model(x)

        for h in hooks:
            h.remove()

        activations = [np.concatenate(s, axis=0) for s in post_stores]
        if self._record_pre:
            self._pre_activations = [np.concatenate(s, axis=0) for s in pre_stores]

        return activations

    def get_pre_activations(self) -> List[np.ndarray]:
        """Return post-linear (pre-norm/activation) outputs per block.

        Only available after calling :meth:`get_activations` with
        ``record_pre_activation=True``.
        """
        if not self._record_pre:
            raise RuntimeError(
                "Pre-activation data is not available. "
                "Construct PyTorchAdapter with record_pre_activation=True."
            )
        if self._pre_activations is None:
            raise RuntimeError("Call get_activations() before get_pre_activations().")
        return self._pre_activations

    def get_weights(self) -> List[np.ndarray]:
        """Return weight matrices W of shape (n_in, n_out) for each block's linear layer."""
        return [
            block["linear"].weight.detach().cpu().numpy().T  # (out, in) → (in, out)
            for block in self._blocks
        ]
