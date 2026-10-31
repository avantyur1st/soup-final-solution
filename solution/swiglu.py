"""Own fused SwiGLU (silu(gate) * up) for the Qwen2 MLP, forward and backward.

Replaces the starter example (which the template explicitly allows: "replace
this file") so the activation runs as OUR kernel. Differences vs the example:

- BLOCK 8192 with num_warps=8 (T4 elementwise sweet spot: 32 fp16 elements per
  lane, fully vectorized 128-bit loads/stores) instead of BLOCK 1024/defaults;
- 64-bit offsets (program_id * BLOCK) so tensors beyond 2^31 elements cannot
  overflow the index math;
- no hidden .to() copies: the autograd Function only makes inputs contiguous
  when they are not already, and the down-stream patch passes contiguous
  GEMM outputs, so in the measured run both copies are skipped;
- fp32 intermediate math (matches PyTorch's opmath_t for half, keeps the
  loss curve inside the official tolerance).

The Qwen2MLP patch is the same class-level pattern as the example: Python
resolves the method on every call, so it works before or after Soup builds
the model, and falls back to the original on CPU or a non-silu config.
"""

import torch
import triton
import triton.language as tl

BLOCK = 8192
NUM_WARPS = 8


@triton.jit
def swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def swiglu_bwd_kernel(grad_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n_elements,
                      BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    # d/dgate silu(gate) = sig + gate*sig*(1-sig) = sig*(1 + gate*(1-sig))
    dup = grad * gate * sig
    dgate = grad * up * sig * (1.0 + gate * (1.0 - sig))
    tl.store(dgate_ptr + offsets, dgate.to(dgate_ptr.dtype.element_ty), mask=mask)
    tl.store(dup_ptr + offsets, dup.to(dup_ptr.dtype.element_ty), mask=mask)


class SwiGLU(torch.autograd.Function):
    """silu(gate) * up: one launch each way instead of three PyTorch ops."""

    @staticmethod
    def forward(ctx, gate, up):
        if gate.dtype != up.dtype:
            up = up.to(gate.dtype)
        if not gate.is_contiguous():
            gate = gate.contiguous()
        if not up.is_contiguous():
            up = up.contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](
            gate, up, out, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        if grad.dtype != gate.dtype:
            grad = grad.to(gate.dtype)
        if not grad.is_contiguous():
            grad = grad.contiguous()
        dgate = torch.empty_like(gate)
        dup = torch.empty_like(up)
        n = gate.numel()
        swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](
            grad, gate, up, dgate, dup, n, BLOCK=BLOCK, num_warps=NUM_WARPS)
        return dgate, dup


def swiglu(gate, up):
    return SwiGLU.apply(gate, up)


def patch_qwen2_mlp():
    """Send Qwen2MLP.forward through the fused kernel. A class patch works
    before or after Soup builds the model: Python looks the method up on
    every call."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda:
            return original(self, x)
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = forward
