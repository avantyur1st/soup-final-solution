"""Own chunked cross-entropy for the Qwen2 LM head (H6 + Proposal 10), cuBLAS-GEMM hybrid.

The stock path materializes the full logits tensor [B*T, V~152k] THREE times
(fp16 logits, fp32 CE copy, fp32 log-softmax saved for backward) — ~5 GB of
peak VRAM at the challenge shapes. Here the logits are computed ONCE by
cuBLAS into an fp16 buffer and ALL the CE math — per-tile log-sum-exp
statistics, target log-prob, per-token entropy, top-1 flag and backward
dlogits — runs in OUR Triton kernels.

Proposal 10 (exp/0025): Active-row filtering in backward.
In SFT batches, only ~41.7% of tokens (`targets >= 0`, the supervised assistant
answer tokens) have non-zero loss gradient `g`; the remaining ~58.3% of tokens
(`targets == -100`, prompt + pad tokens) have `g == 0` and strictly zero
`dlogits` and `d_h`.
By saving ONLY `logits_valid = logits.index_select(0, valid_idx)` at the end of
`ChunkedCE.forward` (which shrinks saved logits from 622 MiB to ~260 MiB and
frees the 622 MiB full logits buffer immediately) and running
`ce_bwd_dlogits_kernel` + `torch.mm(buf, weight)` ONLY on `valid_idx` (~42% of
rows) before scattering into `d_h = torch.zeros(N, H)`, we eliminate ~58% of
the backward `lm_head` cuBLAS GEMM FLOPs and ~360 MiB of saved logits VRAM
while producing bit-identical `d_h`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Elementwise tile for the CE math over the materialized fp16 logits — the
# same shape the backward dlogits kernel uses (measured on the T4: a full
# [N, V] read+write pass costs ~4 ms/step at this tile).
STATS_BLOCK_N = 4096
# Rows per dH GEMM chunk in backward; the dlogits kernel overwrites the
# saved logits chunk in place before each cuBLAS multiply.
BWD_CHUNK_ROWS = 2048
BWD_BLOCK_N = 4096
GRAD_SCALE = 256.0


@triton.jit
def ce_tile_stats_kernel(logits_ptr, m_ptr, s_ptr, zs_ptr, am_ptr, V, n_tiles,
                         BLOCK_N: tl.constexpr):
    """One program: the [BLOCK_N] logits slice of one row.

    Per slice: m = max(l), s = sum exp(l - m), zs = sum exp(l - m) * l,
    am = argmax column (global vocab index). Stats go into
    [n_rows, n_tiles] buffers; the torch fold reduces them per row with
    torch.argmax tie-breaking (lowest index among the maxima).
    """
    row = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < V
    l = tl.load(logits_ptr + row * V + cols, mask=cmask,
                other=float("-inf")).to(tl.float32)
    m = tl.max(l, axis=0)
    e = tl.exp(l - m)  # masked lanes: exp(-inf) = 0
    l_safe = tl.where(cmask, l, 0.0)  # 0 * -inf would be NaN in zs
    out_off = row * n_tiles + pid_n
    tl.store(m_ptr + out_off, m)
    tl.store(s_ptr + out_off, tl.sum(e, axis=0))
    tl.store(zs_ptr + out_off, tl.sum(e * l_safe, axis=0))
    tl.store(am_ptr + out_off,
             tl.argmax(l, axis=0).to(tl.int32) + pid_n * BLOCK_N)


@triton.jit
def ce_bwd_dlogits_kernel(logits_ptr, tgt_ptr, m_ptr, s_ptr, g_ptr, V,
                          SCALE: tl.constexpr, BLOCK_N: tl.constexpr):
    """dlogits = (softmax - onehot(y)) * dL/dtoken, fp16 with GRAD_SCALE.

    One program per row per column tile; writes dlogits over the fp16
    logits buffer in place. dL/d(token CE) = dL/dloss / n_valid — the caller
    folds that into g_ptr.
    """
    row = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cmask = cols < V
    logits = tl.load(logits_ptr + row * V + cols, mask=cmask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + row)
    s = tl.load(s_ptr + row)
    g = tl.load(g_ptr + row)
    y = tl.load(tgt_ptr + row)
    p = tl.exp(logits - m) / s
    onehot = tl.where(cols == y, 1.0, 0.0)
    dl = (p - onehot) * g * SCALE
    tl.store(logits_ptr + row * V + cols, dl.to(logits_ptr.dtype.element_ty), mask=cmask)


def _row_fold(m_buf: torch.Tensor, s_buf: torch.Tensor, am_buf: torch.Tensor):
    """Per-row (m*, s*, global argmax) from per-tile buffers.

    Tiny tensors ([N, n_tiles] fp32), so a torch fold costs microseconds and
    keeps the tie-breaking identical to torch.argmax (lowest index among the
    maxima).
    """
    m_star = m_buf.max(dim=1).values
    scale = torch.exp(m_buf - m_star[:, None])
    s_star = (s_buf * scale).sum(dim=1)
    am_star = torch.where(m_buf == m_star[:, None], am_buf, 1 << 30).min(dim=1).values
    return m_star, s_star, am_star


class ChunkedCE(torch.autograd.Function):
    """Chunked lm-head CE: cuBLAS logits GEMM + own Triton CE math + active-row backward."""

    @staticmethod
    def forward(ctx, h, weight, targets, n_valid):
        N, H = h.shape
        V = weight.shape[0]
        device = h.device
        n_valid_f = float(n_valid)
        ctx.input_dtype = h.dtype
        h = h.to(torch.float16).contiguous()
        weight = weight.to(torch.float16).contiguous()
        targets = targets.contiguous()

        logits = torch.empty(N, V, device=device, dtype=torch.float16)
        torch.mm(h, weight.t(), out=logits)

        n_tiles = triton.cdiv(V, STATS_BLOCK_N)
        m_buf = torch.empty(N, n_tiles, device=device, dtype=torch.float32)
        s_buf = torch.empty_like(m_buf)
        zs_buf = torch.empty_like(m_buf)
        am_buf = torch.empty(N, n_tiles, device=device, dtype=torch.int32)
        ce_tile_stats_kernel[(N, n_tiles)](
            logits, m_buf, s_buf, zs_buf, am_buf, V, n_tiles,
            BLOCK_N=STATS_BLOCK_N, num_warps=8)

        m_star, s_star, am_star = _row_fold(m_buf, s_buf, am_buf)
        lse = m_star + torch.log(s_star)

        ylogit = logits.gather(
            1, targets.clamp(min=0).unsqueeze(1)).squeeze(1).to(torch.float32)

        valid = targets >= 0
        token_loss = torch.where(valid, lse - ylogit, torch.zeros_like(lse))
        scale = torch.exp(m_buf - m_star[:, None])
        token_ent = lse - (zs_buf * scale).sum(dim=1) / s_star
        is_top1 = torch.where(valid, (am_star.to(torch.int64) == targets).float(),
                              torch.zeros(N, device=device))

        # Proposal 10: save only active rows (targets >= 0) for backward so the
        # full [N, V] logits buffer (622 MiB) is freed when forward returns and
        # backward skips the ~58% of prompt/pad rows whose dlogits == 0.
        valid_idx = valid.nonzero(as_tuple=True)[0]
        n_v = valid_idx.numel()
        if n_v == N:
            logits_valid = logits
            tgt_valid = targets
            m_valid = m_star
            s_valid = s_star
        elif n_v > 0:
            logits_valid = logits.index_select(0, valid_idx)
            tgt_valid = targets.index_select(0, valid_idx)
            m_valid = m_star.index_select(0, valid_idx)
            s_valid = s_star.index_select(0, valid_idx)
        else:
            logits_valid = logits[:0]
            tgt_valid = targets[:0]
            m_valid = m_star[:0]
            s_valid = s_star[:0]

        ctx.mark_non_differentiable(token_ent, is_top1)
        ctx.save_for_backward(weight, tgt_valid, logits_valid, m_valid, s_valid, valid_idx)
        ctx.N = N
        ctx.H = H
        ctx.n_valid = n_valid_f
        return token_loss, token_ent, is_top1

    @staticmethod
    def backward(ctx, grad_loss, grad_ent, grad_top1):
        weight, tgt_valid, logits_valid, m_valid, s_valid, valid_idx = ctx.saved_tensors
        N, H = ctx.N, ctx.H
        V = weight.shape[0]
        device = weight.device
        if not torch.is_tensor(grad_loss):
            raise AssertionError("ChunkedCE expects a tensor loss gradient")

        n_v = valid_idx.numel()
        if n_v == 0:
            return torch.zeros(N, H, device=device, dtype=ctx.input_dtype), None, None, None

        if grad_loss.numel() == 1:
            g_valid = torch.full((n_v,), float(grad_loss.item()), device=device, dtype=torch.float32)
        else:
            g_valid = grad_loss.reshape(-1).index_select(0, valid_idx).to(torch.float32).contiguous()

        d_h_valid = torch.empty(n_v, H, device=device, dtype=torch.float16)
        for r0 in range(0, n_v, BWD_CHUNK_ROWS):
            r1 = min(r0 + BWD_CHUNK_ROWS, n_v)
            buf = logits_valid[r0:r1]
            ce_bwd_dlogits_kernel[(r1 - r0, triton.cdiv(V, BWD_BLOCK_N))](
                buf, tgt_valid[r0:r1], m_valid[r0:r1].contiguous(),
                s_valid[r0:r1].contiguous(), g_valid[r0:r1].contiguous(), V,
                SCALE=GRAD_SCALE, BLOCK_N=BWD_BLOCK_N, num_warps=8)
            d_h_valid[r0:r1] = torch.mm(buf, weight) / GRAD_SCALE

        if n_v == N:
            return d_h_valid.to(ctx.input_dtype), None, None, None

        d_h = torch.zeros(N, H, device=device, dtype=ctx.input_dtype)
        d_h.index_copy_(0, valid_idx, d_h_valid.to(ctx.input_dtype))
        return d_h, None, None, None
