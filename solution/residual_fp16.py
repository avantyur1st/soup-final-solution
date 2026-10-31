"""S1 (H16-lite): keep the residual stream in fp16 across the trunk.

Under autocast the stock stack keeps the [B, T, H] residual/hidden tensors in
fp32 end-to-end: the embedding emits fp32, Qwen2RMSNorm promotes back to fp32
(`weight(fp32) * x` type promotion), and every projection then pays an
fp32->fp16 cast kernel per call. This patch casts the embedding output to
fp16 ONCE, so the whole trunk lives in fp16:

- every elementwise residual add runs fp16 (half the read+write traffic);
- the autocast input casts of q/k/v/o projections and of the lm-head GEMM
  become no-ops (the inputs already are fp16);
- all saved-for-backward trunk activations (norm inputs, unpad gather/scatter
  buffers) halve their storage and bandwidth.

Numerical contract:

- the GEMM operands are unchanged: autocast rounded the fp32 norm outputs to
  fp16 at every projection input anyway, so the same fp16 values reach cuBLAS;
- the residual adds now round to fp16 per add (stock accumulated in fp32) —
  this is the one real deviation, watched by the loss-curve and eval-loss
  gates (the H5 precedent, exp/0007, passed with the same class of rounding);
- the own RMSNorm kernel (solution/rmsnorm.py) already computes in fp32
  internally and stores in the input dtype, so an fp16 input yields exactly
  the stock fp16 normalized value; only its guard needed relaxing.
"""

from __future__ import annotations

import torch


def patch_embed_output_fp16():
    """Cast nn.Embedding fp32 CUDA outputs to fp16 (class-level, idempotent).

    embed_tokens is the only embedding in the Qwen2 stack, so this single
    hook puts the trunk entry in fp16 without rewriting Qwen2Model.forward.
    """
    import torch.nn as nn

    original = nn.Embedding.forward
    if getattr(original, "_soup_embed_fp16_patch", False):
        return

    def forward(self, input_ids=None, **kwargs):
        out = original(self, input_ids, **kwargs)
        if out.is_cuda and out.dtype == torch.float32:
            out = out.to(torch.float16)
        return out

    forward._soup_embed_fp16_patch = True
    nn.Embedding.forward = forward
