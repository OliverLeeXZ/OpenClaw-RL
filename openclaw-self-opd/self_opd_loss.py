"""Mixed hint-GRPO + action-OPD loss.

Hint samples:
  - reward carries the normalized group-relative hint advantage
  - teacher_log_probs == rollout_log_probs, so the OPD term is zero

Action samples:
  - reward == 0
  - teacher_log_probs stores log pi_old(a* | s + hint)
  - rollout_log_probs stores log pi_old(a* | s)
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable

import torch

from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss


def self_opd_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    grpo_advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs_list = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    need_entropy_for_loss = args.entropy_coef != 0.0
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=need_entropy_for_loss,
        max_seq_lens=max_seq_lens,
    )
    new_log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)
    old_log_probs = torch.cat(old_log_probs_list, dim=0)

    teacher_log_probs_list = batch.get("teacher_log_probs")
    if teacher_log_probs_list is not None:
        device = new_log_probs.device
        teacher_advantages = torch.cat(
            [t.to(device=device) - o.to(device=device) for t, o in zip(teacher_log_probs_list, old_log_probs_list)],
            dim=0,
        )
    else:
        teacher_advantages = torch.zeros_like(grpo_advantages)

    w_hint = float(os.getenv("OPENCLAW_SELF_OPD_W_HINT", "1.0"))
    w_action = float(os.getenv("OPENCLAW_SELF_OPD_W_ACTION", "1.0"))
    combined_advantages = w_hint * grpo_advantages + w_action * teacher_advantages

    ppo_kl = old_log_probs - new_log_probs
    pg_loss, pg_clipfrac = compute_policy_loss(
        ppo_kl,
        combined_advantages,
        args.eps_clip,
        args.eps_clip_high,
    )
    pg_loss = sum_of_sample_mean(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl_mean = sum_of_sample_mean(ppo_kl)

    if need_entropy_for_loss:
        entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        with torch.no_grad():
            _, ent_data = get_log_probs_and_entropy(
                logits,
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                with_entropy=True,
                max_seq_lens=max_seq_lens,
            )
            entropy_loss = sum_of_sample_mean(torch.cat(ent_data["entropy"], dim=0))

    loss = pg_loss - args.entropy_coef * entropy_loss

    kl_loss = torch.tensor(0.0, device=logits.device)
    if args.use_kl_loss and batch.get("ref_log_probs") is not None:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        kl = compute_approx_kl(
            new_log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    if new_log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_lp = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_lp).abs())

    reported_loss: dict[str, torch.Tensor] = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl_mean.clone().detach(),
    }
    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    return loss, reported_loss
