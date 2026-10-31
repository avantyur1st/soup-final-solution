"""P5 (share keystone): per-use NF4 dequantization in own Triton kernels.

bitsandbytes runs every frozen 4-bit projection through its own CUDA kernels:
a fused `gemm_4bit` in the forward and a separate `dequantize_4bit` in the
backward. None of that time counts toward the own-kernel share, and on a
stack where the own kernels are a thin minority the >=5% share gate is
decided by exactly this work (the 4-bit projections are ~1.3G parameters
revisited fwd+bwd every step).

This module takes the dequant path over: an own `@triton.jit` kernel unpacks
the NF4 codebook values and scales them by the per-block absmax on every use
(forward + backward, ~2 x 196 launches per step), a frozen-weight autograd
Function saves nothing (bnb saves the packed weight reference and recomputes
the same dequant in its backward anyway), and the projections then run as
plain cuBLAS fp16 GEMMs. The dequant->cuBLAS numeric path is the one already
validated end-to-end on the speed lineage; the codebook multiply here is
bit-exact with bitsandbytes` `dequantize_4bit` (verified on the GPU in
`test_kernels.py` and against the bnb CPU backend locally).

Layout per Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs"
(2023): 4-bit NF4 codebook values, absmax per contiguous block of
`quant_state.blocksize` flattened elements, double-quantized absmax
(blockwise 8-bit, blocksize 256, fp32 offset). Two nibbles per byte: the
HIGH nibble is the even element, the LOW nibble the odd one; the final byte
of an odd-numel weight carries one padding nibble that is masked out.

The absmax dequant (numel/blocksize fp32 entries, a few thousand per weight)
is metadata, not weight data: it is computed once per weight and cached
keyed by id(weight) -- LoRA training never replaces parameter objects and
quantization happens once at model build, before any forward. Every
unusual configuration (trainable base weight, trainable bias, CPU, non-fp16
input, missing quant_state) falls back to the stock forward.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_DEQ_BLOCK = 4096
# Frozen per-weight dequant metadata, keyed by id(weight). Same lifetime
# argument as the speed-lineage weight cache: quantization happens once at
# model build, before any forward, and LoRA training never replaces
# parameter objects.
_META_CACHE = {}


@triton.jit
def nf4_dequant_kernel(
    packed_ptr,      # uint8 [ceil(numel/2)], two nibbles per byte
    absmax_ptr,      # fp32 [cdiv(numel, blocksize)]
    code_ptr,        # fp32 [16] NF4 codebook
    out_ptr,         # T [numel]
    numel,
    blocksize,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    # HIGH nibble -> even element, LOW nibble -> odd element.
    byte = tl.load(packed_ptr + (offs // 2), mask=mask, other=0).to(tl.int32)
    shift = tl.where((offs & 1) == 0, 4, 0)
    nib = (byte >> shift) & 0x0F
    val = tl.load(code_ptr + nib, mask=mask, other=0.0)
    scale = tl.load(absmax_ptr + offs // blocksize, mask=mask, other=0.0)
    out = (val * scale).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, out, mask=mask)


def _meta(weight):
    """Frozen per-weight dequant metadata: (absmax fp32, code, blocksize, shape)."""
    meta = _META_CACHE.get(id(weight))
    if meta is None:
        import bitsandbytes.functional as bnb_f

        qs = weight.quant_state
        if qs.nested:
            # Mirrors dequantize_4bit: blockwise-dequant the 8-bit absmax,
            # add the fp32 offset, keep fp32. One-time, a few KB per weight.
            absmax = bnb_f.dequantize_blockwise(qs.absmax, qs.state2)
            absmax = (absmax + qs.offset).float()
        else:
            absmax = qs.absmax.float()
        meta = (absmax.contiguous(), qs.code.float().contiguous(),
                qs.blocksize, tuple(qs.shape))
        _META_CACHE[id(weight)] = meta
    return meta


def _dequant16(packed, meta, dtype):
    """Dequantize one packed NF4 weight to `dtype` with the own kernel."""
    absmax, code, blocksize, shape = meta
    numel = shape[0] * shape[1]
    out = torch.empty(numel, device=packed.device, dtype=dtype)
    grid = (triton.cdiv(numel, _DEQ_BLOCK),)
    nf4_dequant_kernel[grid](
        packed.view(-1), absmax, code, out, numel, blocksize, BLOCK=_DEQ_BLOCK,
    )
    return out.view(shape)


class _NF4LinearFn(torch.autograd.Function):
    """Frozen 4-bit linear with per-use own-kernel dequant and zero saves.

    Weight and bias are frozen, so neither is needed in backward: the
    dequantized weight is recomputed per use (own kernel, counted time) and
    only dx is returned. bnb stock saves the packed weight reference and
    runs the same recompute through its own kernels.
    """

    @staticmethod
    def forward(ctx, x, packed, meta_refs, bias):
        w16 = _dequant16(packed, meta_refs, x.dtype)
        out = torch.nn.functional.linear(x, w16, bias)
        ctx.meta = meta_refs
        ctx.packed = packed
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = None
        if ctx.needs_input_grad[0]:
            w16 = _dequant16(ctx.packed, ctx.meta, grad_out.dtype)
            grad_x = grad_out @ w16
        return grad_x, None, None, None


def patch_linear4bit_own_dequant():
    """Route frozen fp16 4-bit linears through the own-kernel dequant path.

    Idempotent class-level wrapper on bitsandbytes.nn.Linear4bit.forward,
    the same interception point the speed lineage validated under peft
    lora.Linear4bit (the wrapper is reached through base_layer).
    """
    import bitsandbytes as bnb

    original = bnb.nn.Linear4bit.forward
    if getattr(original, "_soup_nf4_dequant_patch", False):
        return

    def forward(self, x, *args, **kwargs):
        weight = getattr(self, "weight", None)
        qs = getattr(weight, "quant_state", None)
        bias = getattr(self, "bias", None)
        if (qs is not None
                and getattr(x, "is_cuda", False)
                and x.dtype == torch.float16
                and not weight.requires_grad
                and (bias is None or not bias.requires_grad)
                and len(qs.shape) == 2
                and not getattr(qs, "packing_format_for_cpu", False)):
            packed = weight.data
            if packed.stride(-1) != 1:
                packed = packed.contiguous()
            meta = _meta(weight)
            return _NF4LinearFn.apply(x, packed, meta, bias)
        return original(self, x, *args, **kwargs)

    forward._soup_nf4_dequant_patch = True
    bnb.nn.Linear4bit.forward = forward
