"""H6: chunked lm-head cross-entropy wired into TRL's SFTTrainer.

TRL 0.29.1 computes the training loss by running the full model forward and
materializing ``outputs.logits`` [B, T, V~152k] — then upcasts to fp32 for
the CE, keeps log-softmax for backward, and re-reads the same logits for the
entropy and accuracy metrics. That tensor is the single biggest VRAM term of
the step and a lot of avoidable memory traffic.

This module replaces ``trl.SFTTrainer.compute_loss`` (the single method both
the plain trainer and soup's ``RewindTrainer`` delegate to): training steps
run the decoder trunk, then our ``ChunkedCE`` (chunked_ce.py) on the flattened
hidden states — logits exist only as register tiles inside our kernels.

Everything else is replicated exactly as TRL 0.29.1 does it:

- loss normalization: ``token_loss.sum() / num_items_in_batch`` when the
  gradient-accumulation fix provides it, else the in-batch mean — identical
  to the ``ForCausalLMLoss`` reduction Trainer triggers with
  ``model_accepts_loss_kwargs=True`` (no extra /grad-accum division, matching
  ``Trainer.training_step``);
- ``entropy``: masked mean of per-position entropy over ``attention_mask``
  (all positions, prompt included), gathered via ``gather_for_metrics``;
- ``mean_token_accuracy``: shifted argmax correctness over
  ``shift_labels != -100``, gathered the same way;
- ``num_tokens``: the ``_total_train_tokens`` accounting is kept;
- rewind: the per-step record is published under soup's
  ``_soup_chunked_step`` contract (``ChunkedStep`` from
  ``soup_cli.utils.stream_loss_chunk``) — token_loss/token_entropy [B, T] with
  the causal shift already applied — so ``RewindTrainer`` and ``soup rewind``
  keep working unchanged, as they do for soup's own chunked steps.

Every configuration the chunked path does not cover — evaluation, liger,
label smoothing, a custom compute_loss_func, aux loss, prompt tuning, soup's
own stream-loss-chunk hook, a quantized or absent lm_head, unexpected input
keys — falls back to the original method, which then behaves byte-for-byte
like stock TRL.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .chunked_ce import ChunkedCE

# one-shot diagnostics printed on the first training step; never affect the
# measurement (a few host-synced scalars and 3 log lines)
_PROBE_DONE = False


def _run_probes(base, inputs):
    global _PROBE_DONE
    _PROBE_DONE = True
    am = inputs.get("attention_mask")
    if am is not None:
        pad_fraction = 1.0 - am.float().mean().item()
        print("[soup-probe] pad_fraction=%.3f real_rows=%d/%d" % (
            pad_fraction, int(am.sum().item()), am.numel()))
    print("[soup-probe] attn_impl=%s sdp_flags flash/mem_eff/math=%s/%s/%s" % (
        getattr(base.config, "_attn_implementation", None),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled()))
    try:
        import torch.nn.attention as _tta
        q = torch.randn(1, 2, 64, 32, device=am.device, dtype=torch.float16)
        with _tta.sdpa_kernel([_tta.SDPBackend.EFFICIENT_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        print("[soup-probe] mem_efficient fp16: OK")
    except Exception as exc:
        print("[soup-probe] mem_efficient fp16: FAIL", type(exc).__name__)

# Inputs the chunked path knows how to run. Anything else in the batch dict
# (vision pixels, CP sharding, ...) means an unusual collator — fall back.
_ALLOWED_INPUT_KEYS = frozenset(
    {"input_ids", "attention_mask", "position_ids", "labels", "use_cache"})


def _shift_targets(labels: torch.Tensor) -> torch.Tensor:
    """The ForCausalLM shift: position t is scored against labels[:, t+1]."""
    return F.pad(labels, (0, 1), value=-100)[..., 1:]


def patch_sft_compute_loss():
    """Route SFTTrainer.compute_loss through the own chunked-CE kernels.

    Idempotent; patches the class once. The replacement degrades to the
    original method on every path it does not fully own, so a silent fallback
    can never change training semantics.
    """
    from trl import SFTTrainer

    if getattr(SFTTrainer.compute_loss, "_soup_chunked_ce_patch", False):
        return

    original = SFTTrainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        chunked_ok = (
            mode == "train"
            and getattr(model, "training", False)
            and "labels" in inputs
            and "shift_labels" not in inputs
            and ("attention_mask" in inputs or "position_ids" in inputs)
            and not getattr(self.args, "use_liger_kernel", False)
            and self.label_smoother is None
            and self.compute_loss_func is None
            and not getattr(self, "aux_loss_enabled", False)
            and getattr(self, "num_virtual_tokens", 0) == 0
            and set(inputs) <= _ALLOWED_INPUT_KEYS
        )
        if chunked_ok:
            # soup's own stream-loss-chunk hook owns the step when installed;
            # never run two chunked losses over one batch.
            from soup_cli.utils.stream_loss_chunk import HOOK_ATTR
            chunked_ok = getattr(self, HOOK_ATTR, None) is None
        if not chunked_ok:
            return original(self, model, inputs,
                            return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)

        from soup_cli.utils.stream_loss_chunk import STEP_ATTR, ChunkedStep

        inputs.pop("_prediction_loss_only", None)
        inputs = self._prepare_inputs(inputs)

        # The trunk is everything before the head; LoRA adapters live on its
        # modules, so calling it directly keeps the adapters active.
        base = self.accelerator.unwrap_model(model)
        if hasattr(base, "get_base_model"):
            base = base.get_base_model()
        trunk = getattr(base, "model", None)
        weight = getattr(getattr(base, "lm_head", None), "weight", None)
        if (trunk is None or weight is None
                or weight.dtype not in (torch.float16, torch.bfloat16,
                                        torch.float32)):
            # unexpected architecture (e.g. a quantized head) — stock path
            return original(self, model, inputs,
                            return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        weight = weight.detach()
        if not _PROBE_DONE:
            _run_probes(base, inputs)

        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask")
        position_ids = inputs.get("position_ids")
        inputs["use_cache"] = False

        # The trunk must run under the same autocast context accelerate wraps
        # the top-level model forward in, so the hidden states are identical
        # to the stock step's.
        with self.accelerator.autocast():
            out = trunk(input_ids=inputs["input_ids"],
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]

        B, T = labels.shape
        H = hidden.shape[-1]
        targets = _shift_targets(labels).reshape(-1)
        labelled = targets >= 0
        if num_items_in_batch is not None:
            n_valid = num_items_in_batch
        else:
            n_valid = labelled.sum()

        token_loss, token_ent, is_top1 = ChunkedCE.apply(
            hidden.reshape(B * T, H), weight, targets, n_valid)

        # Same reduction the model's ForCausalLMLoss performs with
        # num_items_in_batch (sum/NIB) or without it (batch mean).
        loss = token_loss.sum() / n_valid
        if (num_items_in_batch is not None
                and getattr(self.args, "average_tokens_across_devices", False)):
            # Single-process runs see num_processes == 1: this mirrors
            # Trainer.compute_loss without changing the value here.
            loss *= self.accelerator.num_processes

        # Publish the per-token record for the rewind flight recorder
        # (RewindTrainer reads it when the outputs carry no logits).
        token_loss = token_loss.view(B, T)
        token_entropy = token_ent.view(B, T)
        setattr(self, STEP_ATTR, ChunkedStep(
            token_entropy=token_entropy,
            token_loss=token_loss,
            correct=is_top1.sum(),
            labelled=labelled.view(B, T),
        ))

        # Metrics: replicated verbatim from TRL 0.29.1's compute_loss.
        with torch.no_grad():
            if attention_mask is not None:
                entropy = (token_entropy * attention_mask).sum() / attention_mask.sum()
                num_tokens_in_batch = self.accelerator.gather_for_metrics(
                    attention_mask.sum()).sum().item()
            else:
                entropy = token_entropy.mean()
                local_num_tokens = torch.tensor(
                    position_ids.size(1), device=position_ids.device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(
                    local_num_tokens).sum().item()
            entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)

        self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        total_tokens = self.accelerator.gather_for_metrics(labelled.sum())
        correct_tokens = self.accelerator.gather_for_metrics(is_top1.sum())
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        # return_outputs=True is what RewindTrainer passes on every training
        # micro-batch; it reads the chunked record when outputs have no
        # logits, so None is the correct "no logits" output here.
        return (loss, None) if return_outputs else loss

    compute_loss._soup_chunked_ce_patch = True
    SFTTrainer.compute_loss = compute_loss
