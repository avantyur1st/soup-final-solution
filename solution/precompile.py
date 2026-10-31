"""P8: post-import Triton precompile (wall-clock, not step time).

Every own @triton.jit kernel compiles on first use; on this stack that lands
inside the measured `soup train` window, between model load and step 1.
apply() starts a daemon thread that touches every kernel once on small
throwaway tensors while the main thread builds the model, so the measured
training starts with a warm Triton cache. Zero numerics impact.

Spec matching (why small warm tensors hit the production compile keys):
Triton specializes on constexpr values (all fixed inside our launchers) and
on 16-divisibility / ==1 of integer arguments. Every hidden dimension
(1536, 8960, 151936) and every stride derived from them is 16-divisible at
any batch shape, so scalar args keep the same spec key. For the CE tiles
the real tile count is cdiv(151936, 4096) = 38 (generic: not 1, not a
multiple of 16); the warm V = 8208 yields the same generic key for every
tile denominator in use (1024..8192). The only predictable miss is a
hyper-specialized active-row count after unpad, which may still compile
once in-run -- harmless, everything else hits the warm cache.
"""

from __future__ import annotations

import threading
import traceback

import torch

_STARTED = False


def _warm_rmsnorm(device):
    from solution.rmsnorm import rmsnorm

    x = torch.randn(8, 1536, device=device, dtype=torch.float16, requires_grad=True)
    w = torch.randn(1536, device=device, dtype=torch.float32)
    y = rmsnorm(x, w, 1e-6)
    y.sum().backward()


def _warm_swiglu(device):
    from solution.swiglu import swiglu

    n, h = 256, 8960
    gate = torch.randn(n, h, device=device, dtype=torch.float16, requires_grad=True)
    up = torch.randn(n, h, device=device, dtype=torch.float16, requires_grad=True)
    out = swiglu(gate, up)
    out.backward(torch.ones_like(out))


def _warm_rope(device):
    from solution.rope import fused_rope

    B, T, Hq, Hkv, D = 1, 128, 12, 2, 128
    q = torch.randn(B, T, Hq, D, device=device, dtype=torch.float16,
                    requires_grad=True).transpose(1, 2)
    k = torch.randn(B, T, Hkv, D, device=device, dtype=torch.float16,
                    requires_grad=True).transpose(1, 2)
    cos = torch.randn(B, T, D, device=device, dtype=torch.float16)
    sin = torch.randn(B, T, D, device=device, dtype=torch.float16)
    out_q, out_k = fused_rope(q, k, cos, sin)
    torch.autograd.backward(
        (out_q, out_k),
        (torch.ones_like(out_q), torch.ones_like(out_k)))


def _warm_nf4_dequant(device):
    from solution.nf4_dequant import _dequant16

    shape = (8960, 1536)
    numel = shape[0] * shape[1]
    packed = torch.zeros((numel + 1) // 2, dtype=torch.uint8, device=device)
    absmax = torch.ones(numel // 64, dtype=torch.float32, device=device)
    code = torch.zeros(16, dtype=torch.float32, device=device)
    _dequant16(packed, (absmax, code, 64, shape), torch.float16)


def _warm_chunked_ce(device):
    from solution.chunked_ce import ChunkedCE

    # V = 8208: 16-divisible like 151936, and cdiv(V, block) stays a
    # "generic" (not 1, not 16-divisible) tile count for every block size
    # the launchers use, matching the production specialization key.
    N, H, V = 256, 1536, 8208
    h = torch.randn(N, H, device=device, dtype=torch.float16, requires_grad=True)
    weight = torch.randn(V, H, device=device, dtype=torch.float16)
    targets = torch.randint(0, V, (N,), device=device, dtype=torch.long)
    token_loss, _, _ = ChunkedCE.apply(h, weight, targets, N)
    token_loss.sum().backward()


def _warm_all():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    for step in (_warm_rmsnorm, _warm_swiglu, _warm_rope,
                 _warm_nf4_dequant, _warm_chunked_ce):
        try:
            step(device)
        except Exception:
            # A failed warm step must never break the run: the kernel just
            # compiles in-run like it did before this module existed.
            traceback.print_exc()


def start_precompile():
    """Start the background warm thread once, from apply()."""
    global _STARTED
    if _STARTED:
        return
    _STARTED = True
    threading.Thread(target=_warm_all, name="soup-precompile",
                     daemon=True).start()
