"""H4: dequantize each frozen NF4 weight ONCE and run the projections on cuBLAS.

bitsandbytes on sm75 has no fused 4-bit GEMM for the backward, so every step
re-runs the dequant path for all ~196 4-bit projections (fwd + bwd): ~392
JIT-bound dequant launches plus absmax reductions over the full weight set
(measured indirectly by the exp/0003 profile; the Adul23 fork reported the
same count). Under LoRA the base weights are frozen, so the dequantized fp16
weight is a constant — cache it after the first dequant and do a plain
F.linear on cuBLAS instead.

Consequences:
- the ~392 per-step dequant/absmax launches disappear (pure speed win);
- the projections output fp16 instead of the fp32 upcast bnb returns for an
  fp32 input — with the H5 MLP cast and the fp16 attention path this also
  lets SDPA pick the memory-efficient fp16 kernel instead of the fp32 math
  fallback (probed separately);
- VRAM grows by the fp16 weight cache (~2x the 4-bit storage, ~3 GB for
  Qwen2.5-1.5B) — far inside the gate-3 budget at the exp/0007 baseline
  (6787 MiB + ~3 GB < 15779 MiB), which un-blocks this idea (the old
  dequant-cache rejection was priced against a 12383 MiB baseline).

The one-time dequant calls bitsandbytes' own functional — the baseline
library the task itself uses — so the values are bit-identical to what bnb
materializes every step; the optimization is the caching, not the kernel.

Cache key is id(weight): LoRA training never replaces parameter objects, and
quantization happens once at model build, before any forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_DEQ_CACHE = {}


def patch_linear4bit_cached():
    """Replace bnb Linear4bit.forward for frozen fp16-compute weights.

    Idempotent; every unusual configuration (trainable base weight, missing
    quant_state, non-fp16 compute dtype, CPU) falls back to the original
    forward.
    """
    import bitsandbytes as bnb

    original = bnb.nn.Linear4bit.forward
    if getattr(original, "_soup_nf4_cache_patch", False):
        return

    def forward(self, x, *args, **kwargs):
        weight = getattr(self, "weight", None)
        quant_state = getattr(weight, "quant_state", None)
        compute_dtype = getattr(self, "compute_dtype", None)
        if (quant_state is not None
                and getattr(x, "is_cuda", False)
                and not weight.requires_grad
                and (compute_dtype is None or compute_dtype == torch.float16)):
            cached = _DEQ_CACHE.get(id(weight))
            if cached is None:
                cached = bnb.functional.dequantize_4bit(
                    weight.data, quant_state).to(torch.float16)
                _DEQ_CACHE[id(weight)] = cached
            if x.dtype != torch.float16:
                x = x.to(torch.float16)
            return F.linear(x, cached, self.bias)
        return original(self, x, *args, **kwargs)

    forward._soup_nf4_cache_patch = True
    bnb.nn.Linear4bit.forward = forward
