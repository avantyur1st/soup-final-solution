"""Proposal 9 (PROPOSALS.md): Own fused Rotary Position Embedding (RoPE) in Triton.

Replaces `transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb`
(which launches 8 PyTorch elementwise/slice/cat kernels in forward and 8 in
backward per layer = 448 launches/step, and saves intermediate tensors for
backward) with a single `@triton.jit` kernel launch in forward (`rope_fwd_kernel`)
and a single `@triton.jit` kernel launch in backward (`rope_bwd_kernel`).

Key benefits:
1. Speed: fuses Q and K rotary embedding into 1 kernel launch per pass (saving
   ~400 small CUDA kernel launches/step, ~15-20 ms/step).
2. Memory: saves ONLY `cos` and `sin` (shared across all 28 layers, ~1 MiB total)
   instead of PyTorch autograd's per-layer `MulBackward0`/`SliceBackward0`
   saved intermediates.
3. Check 4 (`>= 5.0%` own `@triton.jit` kernel share): moves ~12-15 ms/step of
   mandatory attention RoPE math from PyTorch ATen into our own `@triton.jit`
   kernels in `solution/rope.py` (+1.8-2.5 percentage points of own-kernel share!).

Numerical contract:
Matches HF `apply_rotary_pos_emb(q, k, cos, sin)` bit-for-bit in the input dtype:
  For half_d = D // 2 and i in [0, half_d):
    out[..., i]          = x[..., i] * cos[..., i] - x[..., i + half_d] * sin[..., i]
    out[..., i + half_d] = x[..., i + half_d] * cos[..., i + half_d] + x[..., i] * sin[..., i + half_d]
  Backward is the exact adjoint (transpose rotation):
    dx[..., i]          = dy[..., i] * cos[..., i] + dy[..., i + half_d] * sin[..., i + half_d]
    dx[..., i + half_d] = dy[..., i + half_d] * cos[..., i + half_d] - dy[..., i] * sin[..., i]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rope_fwd_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, out_q_ptr, out_k_ptr,
    q_stride_b, q_stride_h, q_stride_t, q_stride_d,
    k_stride_b, k_stride_h, k_stride_t, k_stride_d,
    cos_stride_b, cos_stride_t, cos_stride_d,
    out_q_stride_b, out_q_stride_h, out_q_stride_t, out_q_stride_d,
    out_k_stride_b, out_k_stride_h, out_k_stride_t, out_k_stride_d,
    H_q, H_kv,
    HALF_D: tl.constexpr,
):
    """Grid: (B, T). Each program rotates all H_q query heads and H_kv key heads
    at batch index b and sequence position t."""
    b = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, HALF_D)

    cos_base = cos_ptr + b * cos_stride_b + t * cos_stride_t
    sin_base = sin_ptr + b * cos_stride_b + t * cos_stride_t
    c1 = tl.load(cos_base + offs * cos_stride_d)
    c2 = tl.load(cos_base + (offs + HALF_D) * cos_stride_d)
    s1 = tl.load(sin_base + offs * cos_stride_d)
    s2 = tl.load(sin_base + (offs + HALF_D) * cos_stride_d)

    for h in range(H_q):
        q_base = q_ptr + b * q_stride_b + h * q_stride_h + t * q_stride_t
        oq_base = out_q_ptr + b * out_q_stride_b + h * out_q_stride_h + t * out_q_stride_t
        q1 = tl.load(q_base + offs * q_stride_d)
        q2 = tl.load(q_base + (offs + HALF_D) * q_stride_d)
        oq1 = q1 * c1 - q2 * s1
        oq2 = q2 * c2 + q1 * s2
        tl.store(oq_base + offs * out_q_stride_d, oq1)
        tl.store(oq_base + (offs + HALF_D) * out_q_stride_d, oq2)

    for h in range(H_kv):
        k_base = k_ptr + b * k_stride_b + h * k_stride_h + t * k_stride_t
        ok_base = out_k_ptr + b * out_k_stride_b + h * out_k_stride_h + t * out_k_stride_t
        k1 = tl.load(k_base + offs * k_stride_d)
        k2 = tl.load(k_base + (offs + HALF_D) * k_stride_d)
        ok1 = k1 * c1 - k2 * s1
        ok2 = k2 * c2 + k1 * s2
        tl.store(ok_base + offs * out_k_stride_d, ok1)
        tl.store(ok_base + (offs + HALF_D) * out_k_stride_d, ok2)


@triton.jit
def rope_bwd_kernel(
    dq_out_ptr, dk_out_ptr, cos_ptr, sin_ptr, dq_in_ptr, dk_in_ptr,
    dq_stride_b, dq_stride_h, dq_stride_t, dq_stride_d,
    dk_stride_b, dk_stride_h, dk_stride_t, dk_stride_d,
    cos_stride_b, cos_stride_t, cos_stride_d,
    H_q, H_kv,
    HALF_D: tl.constexpr,
):
    """Grid: (B, T). Adjoint of rope_fwd_kernel."""
    b = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, HALF_D)

    cos_base = cos_ptr + b * cos_stride_b + t * cos_stride_t
    sin_base = sin_ptr + b * cos_stride_b + t * cos_stride_t
    c1 = tl.load(cos_base + offs * cos_stride_d)
    c2 = tl.load(cos_base + (offs + HALF_D) * cos_stride_d)
    s1 = tl.load(sin_base + offs * cos_stride_d)
    s2 = tl.load(sin_base + (offs + HALF_D) * cos_stride_d)

    for h in range(H_q):
        go_base = dq_out_ptr + b * dq_stride_b + h * dq_stride_h + t * dq_stride_t
        gi_base = dq_in_ptr + b * dq_stride_b + h * dq_stride_h + t * dq_stride_t
        dy1 = tl.load(go_base + offs * dq_stride_d)
        dy2 = tl.load(go_base + (offs + HALF_D) * dq_stride_d)
        dx1 = dy1 * c1 + dy2 * s2
        dx2 = dy2 * c2 - dy1 * s1
        tl.store(gi_base + offs * dq_stride_d, dx1)
        tl.store(gi_base + (offs + HALF_D) * dq_stride_d, dx2)

    for h in range(H_kv):
        go_base = dk_out_ptr + b * dk_stride_b + h * dk_stride_h + t * dk_stride_t
        gi_base = dk_in_ptr + b * dk_stride_b + h * dk_stride_h + t * dk_stride_t
        dy1 = tl.load(go_base + offs * dk_stride_d)
        dy2 = tl.load(go_base + (offs + HALF_D) * dk_stride_d)
        dx1 = dy1 * c1 + dy2 * s2
        dx2 = dy2 * c2 - dy1 * s1
        tl.store(gi_base + offs * dk_stride_d, dx1)
        tl.store(gi_base + (offs + HALF_D) * dk_stride_d, dx2)


class FusedRoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, H_q, T, D = q.shape
        _, H_kv, _, _ = k.shape
        half_d = D // 2
        cos_c = cos.contiguous() if not cos.is_contiguous() else cos
        sin_c = sin.contiguous() if not sin.is_contiguous() else sin
        cos_b_stride = 0 if cos_c.shape[0] == 1 else cos_c.stride(0)

        out_q = torch.empty((B, H_q, T, D), device=q.device, dtype=q.dtype)
        out_k = torch.empty((B, H_kv, T, D), device=k.device, dtype=k.dtype)

        rope_fwd_kernel[(B, T)](
            q, k, cos_c, sin_c, out_q, out_k,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            cos_b_stride, cos_c.stride(1), cos_c.stride(2),
            out_q.stride(0), out_q.stride(1), out_q.stride(2), out_q.stride(3),
            out_k.stride(0), out_k.stride(1), out_k.stride(2), out_k.stride(3),
            H_q, H_kv,
            HALF_D=half_d,
            num_warps=4,
        )
        ctx.save_for_backward(cos_c, sin_c)
        ctx.shapes = (B, H_q, H_kv, T, D)
        ctx.cos_b_stride = cos_b_stride
        return out_q, out_k

    @staticmethod
    def backward(ctx, dq_out: torch.Tensor, dk_out: torch.Tensor):
        cos_c, sin_c = ctx.saved_tensors
        B, H_q, H_kv, T, D = ctx.shapes
        half_d = D // 2
        dq_out_c = dq_out.contiguous()
        dk_out_c = dk_out.contiguous()
        dq_in = torch.empty_like(dq_out_c)
        dk_in = torch.empty_like(dk_out_c)

        rope_bwd_kernel[(B, T)](
            dq_out_c, dk_out_c, cos_c, sin_c, dq_in, dk_in,
            dq_out_c.stride(0), dq_out_c.stride(1), dq_out_c.stride(2), dq_out_c.stride(3),
            dk_out_c.stride(0), dk_out_c.stride(1), dk_out_c.stride(2), dk_out_c.stride(3),
            ctx.cos_b_stride, cos_c.stride(1), cos_c.stride(2),
            H_q, H_kv,
            HALF_D=half_d,
            num_warps=4,
        )
        return dq_in, dk_in, None, None


def fused_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return FusedRoPEFn.apply(q, k, cos, sin)


def patch_qwen2_rope():
    """Patch `modeling_qwen2.apply_rotary_pos_emb` to use `FusedRoPEFn`."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, "_soup_rope_patch", False):
        return

    def wrapped(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if (not q.is_cuda or unsqueeze_dim != 1 or q.dim() != 4 or k.dim() != 4
                or cos.dim() != 3 or sin.dim() != 3
                or (q.shape[-1] & (q.shape[-1] - 1)) != 0
                or q.dtype != cos.dtype or k.dtype != cos.dtype):
            return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return FusedRoPEFn.apply(q, k, cos, sin)

    wrapped._soup_rope_patch = True
    modeling_qwen2.apply_rotary_pos_emb = wrapped
