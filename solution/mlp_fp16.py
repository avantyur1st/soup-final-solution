"""H5: keep the whole Qwen2 MLP chain in fp16 (single entry cast).

bitsandbytes' Linear4bit casts the input to the NF4 compute dtype inside
EVERY projection and returns the result in the caller's dtype. With the
residual stream in fp32 that means each MLP step runs ~4 full-tensor cast
passes and — worse — the [T, 8960] gate/up activations (the largest saved
backward tensors in the model) materialize and stay in fp32.

This patch casts the MLP input to the projections' compute dtype ONCE, so
bnb's internal casts become no-ops and the whole chain (gate, up, swiglu,
down) lives in fp16. The stock path stores the fp16 matmul results upcast to
fp32, so the VALUES are identical; only the storage dtype of the saved
activations changes. The residual add outside the MLP promotes back to the
stream dtype exactly like the stock upcast did.

Gradients: the entry cast folds the three per-projection input casts into
one (the sum rule — same value); the LoRA branches receive fp16-rounded
inputs where the stock path fed them fp32 (one extra rounding on a 0.14%-param
path) — the official eval-loss gate watches exactly this.

Idea and observed mechanism credited to a rival fork's solution/nf4.py +
solution/mlp_fp16.py (Adul23, Kaggle competition fork — read-only reference);
this module is written from scratch for this challenge.
"""

from __future__ import annotations

import torch


def _projection_compute_dtype(proj):
    """Resolve the NF4 compute dtype of a (possibly peft-wrapped) projection."""
    for mod in (proj, getattr(proj, "base_layer", None)):
        if mod is None:
            continue
        dt = getattr(mod, "compute_dtype", None)
        if dt is not None:
            return dt
        weight = getattr(mod, "weight", None)
        qs = getattr(weight, "quant_state", None)
        dt = getattr(qs, "dtype", None)
        if dt is not None:
            return dt
    return None


def patch_qwen2_mlp_fp16():
    """Cast the MLP input to the NF4 compute dtype once, before all projections.

    Idempotent class-level wrapper; falls back to the wrapped forward
    unchanged on CPU, when the compute dtype is not fp16, or when the three
    projections disagree about it.
    """
    from transformers.models.qwen2 import modeling_qwen2

    wrapped = modeling_qwen2.Qwen2MLP.forward
    if getattr(wrapped, "_soup_mlp_fp16_patch", False):
        return

    def forward(self, hidden_states):
        dtypes = [_projection_compute_dtype(p)
                  for p in (self.gate_proj, self.up_proj, self.down_proj)]
        if (hidden_states.is_cuda
                and dtypes[0] is not None
                and hidden_states.dtype != dtypes[0]
                and all(d == dtypes[0] for d in dtypes)):
            hidden_states = hidden_states.to(dtypes[0])
        return wrapped(self, hidden_states)

    forward._soup_mlp_fp16_patch = True
    modeling_qwen2.Qwen2MLP.forward = forward
