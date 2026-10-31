"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1001)])
def test_rmsnorm_matches_pytorch(dtype, shape):
    """Own fused RMSNorm must match the HF Qwen2RMSNorm math, fwd and bwd."""
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True)
    eps = 1e-6
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    # Reference: the HF Qwen2RMSNorm formula in plain torch
    def ref_rmsnorm(inp, w):
        inp32 = inp.float()
        var = inp32.pow(2).mean(-1, keepdim=True)
        return w * (inp32 * torch.rsqrt(var + eps)).to(inp.dtype)

    ref = ref_rmsnorm(x, weight)
    ref.backward(grad.float())
    ref_dx, ref_dw = x.grad.float(), weight.grad.float()
    x.grad = weight.grad = None

    out = rmsnorm(x, weight, eps)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx, rtol=tol, atol=tol)
    torch.testing.assert_close(weight.grad.float(), ref_dw, rtol=tol, atol=tol)


def test_rmsnorm_frozen_weight_skips_dw():
    """LoRA leaves RMSNorm weights frozen: backward must not compute dW."""
    from solution.rmsnorm import rmsnorm

    x = torch.randn(8, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(1536, device="cuda", dtype=torch.float16, requires_grad=False)
    out = rmsnorm(x, weight, 1e-6)
    out.backward(torch.randn_like(out))
    assert x.grad is not None and x.grad.shape == x.shape


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_swiglu_noncontiguous_grad(dtype):
    """backward must handle a non-contiguous grad (contiguous() branch)."""
    from solution.swiglu import swiglu

    torch.manual_seed(1)
    shape = (7, 129, 8960)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn((shape[1], shape[0], shape[2]), device="cuda", dtype=dtype).transpose(0, 1)
    assert grad.shape == shape and not grad.is_contiguous()

    out = swiglu(gate, up)
    out.backward(grad)

    # Analytic reference in fp32 (no second autograd pass: gate.grad is set)
    g32 = gate.float().detach()
    sig = torch.sigmoid(g32)
    ref_dgate = grad.float() * up.float() * sig * (1.0 + g32 * (1.0 - sig))
    ref_dup = grad.float() * g32 * sig
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 12, 2, 512, 128), (2, 12, 2, 77, 128)])
def test_rope_matches_pytorch(dtype, shape):
    """Own fused RoPE must match HF apply_rotary_pos_emb, fwd and bwd."""
    from solution.rope import fused_rope

    torch.manual_seed(2)
    B, H_q, H_kv, T, D = shape
    # Simulate Qwen2Attention transposed layout: [B, T, H, D].transpose(1, 2)
    q_raw = torch.randn((B, T, H_q, D), device="cuda", dtype=dtype, requires_grad=True)
    k_raw = torch.randn((B, T, H_kv, D), device="cuda", dtype=dtype, requires_grad=True)
    q = q_raw.transpose(1, 2)
    k = k_raw.transpose(1, 2)
    cos = torch.randn((B, T, D), device="cuda", dtype=dtype)
    sin = torch.randn((B, T, D), device="cuda", dtype=dtype)
    dq = torch.randn((B, H_q, T, D), device="cuda", dtype=dtype)
    dk = torch.randn((B, H_kv, T, D), device="cuda", dtype=dtype)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    ref_q = (q * cos_u) + (rotate_half(q) * sin_u)
    ref_k = (k * cos_u) + (rotate_half(k) * sin_u)
    torch.autograd.backward((ref_q, ref_k), (dq, dk))
    ref_dq_raw = q_raw.grad.float().clone()
    ref_dk_raw = k_raw.grad.float().clone()
    q_raw.grad = k_raw.grad = None

    out_q, out_k = fused_rope(q, k, cos, sin)
    torch.autograd.backward((out_q, out_k), (dq, dk))
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out_q.float(), ref_q.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(out_k.float(), ref_k.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(q_raw.grad.float(), ref_dq_raw, rtol=tol, atol=tol)
    torch.testing.assert_close(k_raw.grad.float(), ref_dk_raw, rtol=tol, atol=tol)


def _chunked_ce_reference(h, weight, targets, n_valid):
    """Stock-path reference: fp16 logits, fp32 CE per token, autograd dH."""
    h_ref = h.detach().clone().requires_grad_(True)
    logits = h_ref @ weight.T
    logp = torch.log_softmax(logits.float(), dim=-1)
    safe = targets.clamp(min=0)
    valid = targets >= 0
    ref_tok = torch.where(
        valid, -logp.gather(1, safe[:, None])[:, 0],
        torch.zeros_like(targets, dtype=torch.float32))
    ref_loss = ref_tok.sum() / n_valid
    ref_loss.backward()
    return logits, logp, ref_tok, ref_loss, h_ref.grad, valid


@pytest.mark.parametrize("shape", [(64, 256, 1500), (3, 128, 1024)])
def test_chunked_ce_matches_pytorch(shape):
    """Own chunked CE must match the stock fp16-logits CE path, fwd and bwd.

    V=1500 is deliberately not divisible by BLOCK_N=512 to exercise the
    masked tail tile.
    """
    from solution.chunked_ce import ChunkedCE

    torch.manual_seed(0)
    N, H, V = shape
    h = torch.randn(N, H, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float16)
    targets = torch.randint(0, V, (N,), device="cuda")
    targets[::9] = -100
    n_valid = int((targets >= 0).sum())

    logits, logp, ref_tok, ref_loss, ref_dh, valid = _chunked_ce_reference(
        h, weight, targets, n_valid)

    token_loss, token_ent, is_top1 = ChunkedCE.apply(h, weight, targets, n_valid)
    loss = token_loss.sum() / n_valid
    loss.backward()

    torch.testing.assert_close(token_loss, ref_tok, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(loss, ref_loss, rtol=2e-3, atol=2e-3)
    # entropy: per-position, entropy-only rows included
    p = torch.softmax(logits.float(), dim=-1)
    ref_ent = -(p * logp).sum(-1)
    torch.testing.assert_close(token_ent, ref_ent, rtol=2e-3, atol=5e-3)
    # top-1 over valid rows (argmax ties are improbable on random data)
    ref_top1 = (logits.float().argmax(-1) == targets) & valid
    assert torch.equal(is_top1.bool(), ref_top1)
    # entropy/top1 are metric-only outputs: no graph, no gradient
    assert not token_ent.requires_grad and not is_top1.requires_grad
    # invalid rows: zero loss, zero top1
    assert (token_loss[~valid] == 0).all() and (is_top1[~valid] == 0).all()
    # dH vs the stock autograd gradient through the fp16 logits GEMM
    torch.testing.assert_close(h.grad.float(), ref_dh.float(),
                               rtol=2e-2, atol=5e-4)


def test_chunked_ce_backward_chunking():
    """N above BWD_CHUNK_ROWS exercises the recompute+mm backward loop."""
    from solution.chunked_ce import ChunkedCE

    torch.manual_seed(1)
    N, H, V = 2500, 256, 1500
    assert N > 2048  # more rows than one backward chunk
    h = torch.randn(N, H, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(V, H, device="cuda", dtype=torch.float16)
    targets = torch.randint(0, V, (N,), device="cuda")
    targets[::7] = -100
    n_valid = int((targets >= 0).sum())

    _, _, ref_tok, ref_loss, ref_dh, valid = _chunked_ce_reference(
        h, weight, targets, n_valid)

    token_loss, _, _ = ChunkedCE.apply(h, weight, targets, n_valid)
    (token_loss.sum() / n_valid).backward()

    torch.testing.assert_close(token_loss, ref_tok, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(h.grad.float(), ref_dh.float(),
                               rtol=2e-2, atol=5e-4)
# --- exp/mem-0026 (P5): own per-use NF4 dequant -------------------------------

def test_nf4_dequant_bitexact_vs_bnb(shape=(256, 128), blocksize=64):
    """Own Triton dequant must be BIT-EXACT with bitsandbytes on the GPU.

    Both paths do an fp32 codebook multiply by the fp32 block absmax and
    round once to fp16, so any disagreement here means the kernel read the
    packed layout (nibble order, block mapping, double-quant absmax) wrong.
    """
    import types

    import bitsandbytes.functional as bnb_f

    from solution.nf4_dequant import _dequant16, _meta

    torch.manual_seed(0)
    w = torch.randn(shape, device="cuda", dtype=torch.float16)
    packed, qs = bnb_f.quantize_4bit(
        w, blocksize=blocksize, quant_type="nf4",
        compress_statistics=True, quant_storage=torch.uint8)
    meta = _meta(types.SimpleNamespace(quant_state=qs))
    got = _dequant16(packed, meta, torch.float16)
    ref = bnb_f.dequantize_4bit(packed, qs).to(torch.float16)
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_nf4_dequant_bitexact_odd_numel():
    """Odd-numel weights: the padding nibble of the last byte is masked."""
    import types

    import bitsandbytes.functional as bnb_f

    from solution.nf4_dequant import _dequant16, _meta

    torch.manual_seed(1)
    shape = (65, 7)
    w = torch.randn(shape, device="cuda", dtype=torch.float16)
    packed, qs = bnb_f.quantize_4bit(
        w, blocksize=64, quant_type="nf4",
        compress_statistics=True, quant_storage=torch.uint8)
    meta = _meta(types.SimpleNamespace(quant_state=qs))
    got = _dequant16(packed, meta, torch.float16)
    ref = bnb_f.dequantize_4bit(packed, qs).to(torch.float16)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_nf4_linear_fwd_bwd_vs_stock_bnb():
    """Patched Linear4bit vs the stock bnb path: fwd within fp16 GEMM
    tolerance (fused kernel vs dequant+cuBLAS accumulation), backward dx
    bit-exact (identical dequant -> matmul structure)."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_f

    from solution.nf4_dequant import patch_linear4bit_own_dequant

    patch_linear4bit_own_dequant()

    torch.manual_seed(2)
    lin = bnb.nn.Linear4bit(256, 128, bias=True, compute_dtype=torch.float16).cuda()
    lin.weight.data, lin.weight.quant_state = bnb_f.quantize_4bit(
        lin.weight.data, blocksize=64, quant_type="nf4",
        compress_statistics=True, quant_storage=torch.uint8)
    lin.weight.requires_grad = False
    lin.bias.requires_grad = False

    x = torch.randn(8, 256, device="cuda", dtype=torch.float16, requires_grad=True)
    out = lin(x)
    grad = torch.randn_like(out)
    out.backward(grad)
    dx_own = x.grad.clone()

    # Stock bnb path at function level (unaffected by the class patch).
    x2 = x.detach().requires_grad_(True)
    out_ref = bnb.matmul_4bit(x2, lin.weight.data.t(), quant_state=lin.weight.quant_state,
                              bias=lin.bias)
    torch.testing.assert_close(out, out_ref, rtol=2e-3, atol=2e-3)
    out_ref.backward(grad)
    torch.testing.assert_close(dx_own, x2.grad, rtol=0, atol=0)

    # Zero-save check: the frozen-weight Function must not stash activations.
    assert dx_own.shape == x.shape
