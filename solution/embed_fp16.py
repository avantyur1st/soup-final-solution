"""M5 (exp/0020, adapted for S1/P1 fp16 residual stream): store the frozen tied
embedding in fp16 — losslessly.

`prepare_model_for_kbit_training` upcasts every non-quantized parameter to
fp32; the tied embedding (890 MiB fp32 at the challenge shapes) is one of
them. Its values originate from the fp16 checkpoint, so the upcast is
reversible without loss: we verify element-wise (row-chunked, to bound the
transient) that `weight.half().float() == weight`, and only then store the
weight in fp16 (~445 MiB of persistent VRAM saved across the entire run).

On the Speed Track (`exp/0024`, S1/P1), the residual stream runs in fp16
starting right after `embed_tokens`. With `embed_tokens.weight` stored in fp16:
1. `embed_tokens` outputs fp16 natively with zero cast kernels (making the
   `residual_fp16.py` hook a no-op).
2. Because `lm_head.weight` is tied to `embed_tokens.weight`, `weight.to(torch.float16)`
   in `ChunkedCE.forward` becomes a zero-cost no-op on all 64 steps instead of
   casting 151936 x 1536 elements (890 MiB -> 445 MiB) on every step!

Guards: only when the weight is frozen (`requires_grad` False), fp32, and the
lossless check passes for the WHOLE tensor; otherwise the module is a no-op
with a printed reason, never an error path.
"""

from __future__ import annotations

import torch


def _losslessly_fp16(weight: torch.Tensor, rows: int = 65536) -> bool:
    """True iff fp32 -> fp16 -> fp32 is the identity on the whole tensor."""
    if weight.dtype != torch.float32:
        return False
    for r0 in range(0, weight.shape[0], rows):
        chunk = weight[r0:r0 + rows]
        if not torch.equal(chunk.to(torch.float16).to(torch.float32), chunk):
            return False
    return True


def _convert(model) -> None:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    embed = base.get_input_embeddings()
    if embed is None:
        print("[soup-embed16] no input embedding found - keeping fp32")
        return
    weight = embed.weight
    if weight.requires_grad:
        print("[soup-embed16] embedding is trainable - keeping fp32")
        return
    if not _losslessly_fp16(weight.data):
        print("[soup-embed16] embedding values are not exactly representable "
              "in fp16 - keeping fp32")
        return
    weight.data = weight.data.to(torch.float16)
    print(f"[soup-embed16] embedding stored fp16 "
          f"({weight.numel() * 2 / 2**20:.0f} MiB saved)")


def patch_prepare_model_for_kbit_training():
    """Wrap peft.prepare_model_for_kbit_training so the conversion runs right
    after the fp32 upcast it performs (soup imports the symbol lazily inside
    the setup method, so patching the module attribute is sufficient)."""
    import peft
    import peft.utils.other

    orig = peft.utils.other.prepare_model_for_kbit_training
    if getattr(orig, "_soup_embed_fp16_patch", False):
        return

    def wrapper(model, *args, **kwargs):
        converted = orig(model, *args, **kwargs)
        _convert(converted if converted is not None else model)
        return converted

    wrapper._soup_embed_fp16_patch = True
    peft.utils.other.prepare_model_for_kbit_training = wrapper
    if getattr(peft, "prepare_model_for_kbit_training", None) is orig:
        peft.prepare_model_for_kbit_training = wrapper
