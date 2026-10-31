"""H7: run the position-wise (norm + MLP) part of each layer on real tokens only.

With right padding, ~25% of the rows in every alpaca batch are pad tokens
(measured: pad_fraction=0.254 on the official run, exp/0008 probe). The stock
layer runs input_layernorm, post_attention_layernorm and the whole MLP on all
B*T rows; pad rows carry garbage embeddings, produce no loss (labels -100)
and, with right padding, are never attended to by real rows (causal mask:
pad positions come after all real tokens of their sequence).

This patch gathers the real rows ONCE per model forward (2D attention_mask,
flat [B*T] index), runs self-attention on the FULL padded batch (attention is
the only op that mixes positions), and executes
post_attention_layernorm + MLP on the compressed [n_real, H] batch, then
scatters back with index_copy_. Pad rows keep their post-attention value
instead of residual + MLP(garbage) — that difference stays confined to pad
positions and never reaches the loss, metrics, rewind log or real-row
gradients (identical to the stock run, where autograd produces zero gradient
for pad rows). exp/0024 (P1): the original implementation violated this
contract — index_copy left pad rows holding residual and the final residual
add doubled them every layer; the fix writes residual + mlp for real rows
into a clone so pad rows pass through exactly once.

Recommendation source: requirements.md §5.2 ("удаление pad-токенов в пределах
батча"); mechanism credit: rival fork Adul23 solution/unpad.py (read-only
reference) — this module is written from scratch; attention stays full as in
that design.

The index is stashed by a Qwen2Model.forward wrapper right before the layer
loop reads it (single-threaded trainer, one model forward per step).
"""

from __future__ import annotations

import torch

# Flat [B*T] index of real (non-pad) rows for the CURRENT model forward.
# None means "no padding information / nothing to compress".
_UNPAD_IDX = None


def patch_qwen2_unpad_mlp():
    """Wrap Qwen2Model.forward (index stash) and Qwen2DecoderLayer.forward
    (gather norm+MLP on real rows). Idempotent; falls back to the original
    forward whenever the mask is missing, all rows are real, or the batch is
    on CPU."""
    from transformers.models.qwen2 import modeling_qwen2

    global _UNPAD_IDX

    model_forward = modeling_qwen2.Qwen2Model.forward
    if getattr(model_forward, "_soup_unpad_patch", False):
        return

    def model_wrapped(self, input_ids=None, attention_mask=None,
                      position_ids=None, past_key_values=None, **kwargs):
        global _UNPAD_IDX
        idx = None
        if (attention_mask is not None and attention_mask.dim() == 2
                and attention_mask.is_cuda):
            idx = attention_mask.reshape(-1).nonzero(as_tuple=True)[0]
            if idx.numel() == attention_mask.numel():
                idx = None  # nothing to compress this step
        _UNPAD_IDX = idx
        try:
            return model_forward(self, input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 past_key_values=past_key_values, **kwargs)
        finally:
            _UNPAD_IDX = None

    model_wrapped._soup_unpad_patch = True
    modeling_qwen2.Qwen2Model.forward = model_wrapped

    layer_forward = modeling_qwen2.Qwen2DecoderLayer.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, position_embeddings=None,
                **kwargs):
        idx = _UNPAD_IDX
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention (unchanged stock sequence, full padded batch)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        if (idx is not None and hidden_states.is_cuda
                and idx.numel() < hidden_states.shape[0] * hidden_states.shape[1]):
            B, T, H = hidden_states.shape
            flat = hidden_states.reshape(B * T, H)
            real_in = flat.index_select(0, idx)
            real = self.post_attention_layernorm(real_in)
            real = self.mlp(real)
            # exp/0024 (P1) pass-through fix: the previous index_copy left
            # pad rows holding residual, and the final residual add below
            # then doubled them EVERY layer (2^k growth — fp32 hid ~1e9,
            # fp16 overflowed at layer 13; measured exp/0022/0023). Real
            # rows must be residual + mlp exactly like stock; pad rows must
            # pass through the post-attention value EXACTLY ONCE (the
            # intended H7 semantics). Addition order matches stock, so the
            # real rows stay bit-identical.
            # H5/H4 make the MLP return fp16 while the stream may be fp32 —
            # cast here, exactly the cast the stock bnb path did.
            out = flat.clone()
            out.index_copy_(0, idx, (real_in + real).to(out.dtype))
            return out.view(B, T, H)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    modeling_qwen2.Qwen2DecoderLayer.forward = forward
