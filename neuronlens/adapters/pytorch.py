"""PyTorch model adapter with block-level detection.

Groups sequences of Linear/Conv2d → [Norm] → [Activation] into display blocks.
Activations captured are post-block (post-activation) by default.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base import ModelAdapter


# ── Layer-type classification ─────────────────────────────────────────────────

def _norm_types():
    import torch.nn as nn
    return (
        nn.BatchNorm1d, nn.BatchNorm2d,
        nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d,
    )


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
    """Walk model.named_modules() and group into Linear- or Conv2d-headed blocks.

    A block is: nn.Linear/Conv2d → [zero or more Norms] → [zero or more Activations].
    The canonical order expected is Conv/Linear → Norm → Activation.
    Norms that appear after an activation are not added to the current block.
    Unknown modules (Dropout, Flatten, MaxPool2d, etc.) are skipped without
    closing the current block, so they remain transparent to block grouping.
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
                "kind":        "linear",
                "linear":      mod,
                "linear_name": name,
                "norms":       [],
                "activations": [],
            }
        elif isinstance(mod, nn.Conv2d):
            if current is not None:
                blocks.append(current)
            current = {
                "kind":      "conv",
                "conv":      mod,
                "conv_name": name,
                "norms":     [],
                "activations": [],
                "spatial":   None,  # (C, H, W) filled during get_activations()
            }
        elif isinstance(mod, norm_t) and current is not None and not current["activations"]:
            # Norms are grouped only before any activation (canonical order)
            current["norms"].append(mod)
        elif isinstance(mod, act_t) and current is not None:
            current["activations"].append(mod)
        # All other modules (MaxPool, Dropout, Flatten, etc.) are skipped

    if current is not None:
        blocks.append(current)

    return blocks


def _block_label(block: dict) -> str:
    if block["kind"] == "conv":
        conv = block["conv"]
        k = conv.kernel_size
        if isinstance(k, (list, tuple)):
            ks = "\u00d7".join(str(x) for x in k)  # "×"
        else:
            ks = f"{k}\u00d7{k}"
        parts = [f"Conv2d({conv.in_channels}\u2192{conv.out_channels}, {ks})"]
    else:
        lin = block["linear"]
        parts = [f"Linear({lin.in_features}\u2192{lin.out_features})"]

    for m in block["norms"]:
        parts.append(type(m).__name__)
    for m in block["activations"]:
        parts.append(type(m).__name__)
    return " \u00b7 ".join(parts)  # " · ".join(parts)


def _block_type(block: dict) -> str:
    kind     = block["kind"]  # "linear" or "conv"
    has_norm = bool(block["norms"])
    has_act  = bool(block["activations"])
    if has_norm and has_act:
        return f"{kind}+norm+activation"
    if has_norm:
        return f"{kind}+norm"
    if has_act:
        return f"{kind}+activation"
    return kind


def _last_module(block: dict):
    """The last module in execution order (used for post-activation hooking)."""
    if block["activations"]:
        return block["activations"][-1]
    if block["norms"]:
        return block["norms"][-1]
    return block.get("linear") or block.get("conv")


def _make_hook(storage: list):
    def hook(module, inp, output):
        storage.append(output.detach().cpu().numpy())
    return hook


# ── Adapter ───────────────────────────────────────────────────────────────────

class PyTorchAdapter(ModelAdapter):
    """Adapter for PyTorch models with block-level detection.

    Groups ``Linear/Conv2d → [Norm] → [Activation]`` sequences into blocks.
    Activations captured are post-block (post-activation) by default.

    Args:
        model: An ``nn.Module`` (typically ``nn.Sequential`` or any module
               whose children can be walked with ``named_modules()``).
        record_pre_activation: If ``True``, also captures post-linear/conv
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
            raise ValueError("No nn.Linear or nn.Conv2d layers found in the model.")

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def get_layers(self) -> List[Dict[str, Any]]:
        """Return one info dict per block.

        Keys: ``name``, ``type``, ``block_label``, ``block_type``.
        Conv blocks additionally include ``channels``, ``spatial_h``, ``spatial_w``
        after :meth:`get_activations` has been called.
        """
        layers = []
        for block in self._blocks:
            info: Dict[str, Any] = {
                "name":        block.get("conv_name", block.get("linear_name")),
                "type":        block["kind"],
                "block_label": _block_label(block),
                "block_type":  _block_type(block),
            }
            if block["kind"] == "conv" and block.get("spatial") is not None:
                C, H, W = block["spatial"]
                info["channels"]  = C
                info["spatial_h"] = H
                info["spatial_w"] = W
            layers.append(info)
        return layers

    def get_activations(self, inputs: np.ndarray) -> List[np.ndarray]:
        """Run inputs and return post-block (post-activation) outputs per block.

        For linear blocks the output shape is ``(n_samples, n_neurons)``.
        For conv blocks the output shape is ``(n_samples, C, H, W)``.
        The spatial dimensions ``(C, H, W)`` are also stored in the block dict
        so that :meth:`get_layers` can expose them after this call.
        """
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
            # Pre-activation: hook on the linear/conv layer (post-linear/conv)
            if self._record_pre:
                hooks.append(
                    (block.get("linear") or block.get("conv")).register_forward_hook(
                        _make_hook(pre_stores[i])
                    )
                )

        # Move input to the same device as the model
        try:
            device = next(self._model.parameters()).device
        except StopIteration:
            device = self._torch.device("cpu")
        x = self._torch.tensor(inputs, dtype=self._torch.float32).to(device)
        with self._torch.no_grad():
            self._model(x)

        for h in hooks:
            h.remove()

        activations = []
        for i, block in enumerate(self._blocks):
            arr = np.concatenate(post_stores[i], axis=0)
            if block["kind"] == "conv":
                # arr shape: (batch, C, H, W) — store spatial dims for get_layers()
                block["spatial"] = arr.shape[1:]  # (C, H, W)
            activations.append(arr)

        if self._record_pre:
            self._pre_activations = [np.concatenate(s, axis=0) for s in pre_stores]

        return activations

    def get_pre_activations(self) -> List[np.ndarray]:
        """Return post-linear/conv (pre-norm/activation) outputs per block.

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
        """Return weight matrices W of shape (n_in, n_out) for each block.

        ``n_in`` is the number of display units in the preceding layer and
        ``n_out`` is the number in the current block.

        - **Linear blocks**: ``linear.weight.T`` → ``(in_features, out_features)``
        - **Conv blocks**: sum of |weight| over spatial kernel →
          ``(in_channels, out_channels)``
        - **Conv→Linear** (after Flatten): group linear weights by input channel →
          ``(in_channels, out_features)``
        """
        weights = []
        for i, block in enumerate(self._blocks):
            if block["kind"] == "conv":
                # conv.weight: (out_C, in_C, kH, kW) → sum over spatial → (in_C, out_C)
                W = block["conv"].weight.detach().cpu().numpy()
                W_reduced = np.abs(W).sum(axis=(2, 3))  # (out_C, in_C)
                weights.append(W_reduced.T)              # (in_C, out_C)

            else:  # linear
                W = block["linear"].weight.detach().cpu().numpy()  # (out, in)

                # Check if the previous block was a conv block (Flatten in between)
                prev_block = self._blocks[i - 1] if i > 0 else None
                if prev_block is not None and prev_block["kind"] == "conv":
                    spatial = prev_block.get("spatial")  # (C, H, W) from forward pass
                    if spatial is not None:
                        C, H, Wd = spatial
                        hw = H * Wd
                        if W.shape[1] == C * hw:
                            # Sum over channels for each spatial position.
                            # Returns (H*W, out) so hover signal propagates
                            # per spatial cell rather than per channel.
                            W_abs = np.abs(W)                          # (out, C*H*W)
                            W_abs = W_abs.reshape(W.shape[0], C, hw)  # (out, C, H*W)
                            W_spatial = W_abs.sum(axis=1)             # (out, H*W)
                            weights.append(W_spatial.T)               # (H*W, out)
                            continue
                    # Fallback: spatial dims unknown or size mismatch
                    C_prev = prev_block["conv"].out_channels
                    out_f  = block["linear"].out_features
                    weights.append(np.ones((C_prev, out_f)) / max(C_prev, 1))
                    continue

                weights.append(W.T)  # (in, out)

        return weights
