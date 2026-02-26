import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

def train_step(
    model,
    opt,
    scaler,
    batch,
    use_amp: bool,
    train_phase: str = "trick",
    hidden_loss_weight: float = 0.3,
    impossible_penalty_weight: float = 2.0,
    forced_imitation_weight: float = 0.5,
) -> dict:
    model.train()
    with autocast(device_type='cuda', enabled=use_amp):
        logits, card_logits, pts_pred, value_pred = model({
            "obs_a":        batch["obs_a"],
            "token_ids":    batch["token_ids"],
            "token_mask":   batch["token_mask"],
            "action_feats": batch["action_feats"],
            "action_mask":  batch["action_mask"],
        })
        # Mask logits *before* softmax so -inf doesn't corrupt the denominator
        # IMPORTANT: We use -1e4 instead of -1e9 because PyTorch AMP (Float16) 
        # has a minimum representable value of roughly -65504.
        masked_logits = logits.masked_fill(batch["action_mask"], -1e4)
        log_p = F.log_softmax(masked_logits, dim=-1)
        chosen_lp = log_p.gather(1, batch["action_idx"].unsqueeze(1)).squeeze(1)
        
        # Policy losses:
        # - PPO ratio path only for model-sampled actions.
        # - Forced heuristic actions use imitation-only loss.
        safe_log_prob_old = torch.clamp(batch["log_prob_old"], min=-100.0)
        chosen_lp_clamped = torch.clamp(chosen_lp, min=-100.0)
        ratio_all = torch.exp(chosen_lp_clamped - safe_log_prob_old)
        adv_all = batch["advantage"]
        is_forced = batch.get("is_forced", torch.zeros_like(chosen_lp)).float() > 0.5
        sampled_mask = ~is_forced
        
        # Value Loss
        value_loss = F.mse_loss(value_pred, batch["value"])
        
        # Entropy Bonus
        probs = F.softmax(logits, dim=-1)
        safe_log_p = log_p.masked_fill(batch["action_mask"], 0.0)
        entropy = -(probs * safe_log_p).sum(dim=-1).mean()
        entropy_loss = -0.01 * entropy
        
        # Points Prediction Loss
        pts_loss = F.mse_loss(pts_pred, batch["pts_target"]) * 0.3

        # Bid Masking Regularizer:
        # Check if the chosen action is a BID (action_type 1 is one-hot at index 1 of action_feats)
        chosen_feats = batch["action_feats"].gather(1, batch["action_idx"].unsqueeze(1).unsqueeze(2).expand(-1, -1, 51)).squeeze(1)
        is_bid = chosen_feats[:, 1] > 0.5

        policy_mask = sampled_mask.clone()
        imitation_mask = is_forced.clone()
        if train_phase == "bidding_value":
            non_bid = ~is_bid
            policy_mask = policy_mask & non_bid
            imitation_mask = imitation_mask & non_bid

        clip_epsilon = 0.2
        ppo_loss = torch.tensor(0.0, device=logits.device)
        approx_kl = torch.tensor(0.0, device=logits.device)
        clipfrac = torch.tensor(0.0, device=logits.device)
        if policy_mask.any():
            adv = adv_all[policy_mask]
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = ratio_all[policy_mask]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
            ppo_loss = -torch.min(surr1, surr2).mean()
            approx_kl = (safe_log_prob_old[policy_mask] - chosen_lp_clamped[policy_mask]).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_epsilon).float().mean()

        forced_imitation_loss = torch.tensor(0.0, device=logits.device)
        if imitation_mask.any():
            forced_imitation_loss = (-chosen_lp[imitation_mask]).mean()

        policy_loss = ppo_loss + forced_imitation_weight * forced_imitation_loss

        # Hidden-state auxiliary loss:
        # - Positive supervision for cards that are truly in opponent hands.
        # - Extra penalty for cards that are symbolically impossible.
        # - No penalty on uncertain negatives (possible-but-not-true cards).
        hidden_target = batch.get("hidden_target")
        hidden_possible = batch.get("hidden_possible")
        hidden_loss = torch.tensor(0.0, device=logits.device)
        hidden_pos_loss = torch.tensor(0.0, device=logits.device)
        hidden_impossible_loss = torch.tensor(0.0, device=logits.device)
        hidden_pos_acc = torch.tensor(0.0, device=logits.device)
        impossible_mass = torch.tensor(0.0, device=logits.device)
        if hidden_target is not None and hidden_possible is not None:
            hidden_target = hidden_target.float()
            hidden_possible = hidden_possible.float()
            pos_mask = hidden_target > 0.5
            impossible_mask = (hidden_possible < 0.5) & (~pos_mask)

            bce_all = F.binary_cross_entropy_with_logits(
                card_logits, hidden_target, reduction="none"
            )

            pos_count = pos_mask.float().sum()
            if pos_count > 0:
                pos_loss = (bce_all * pos_mask.float()).sum() / pos_count
                hidden_pos_loss = pos_loss
                hidden_probs = torch.sigmoid(card_logits)
                hidden_pos_acc = (hidden_probs[pos_mask] > 0.5).float().mean()
            else:
                pos_loss = torch.tensor(0.0, device=logits.device)

            impossible_count = impossible_mask.float().sum()
            if impossible_count > 0:
                impossible_loss = (
                    F.binary_cross_entropy_with_logits(
                        card_logits,
                        torch.zeros_like(card_logits),
                        reduction="none",
                    ) * impossible_mask.float()
                ).sum() / impossible_count
                hidden_impossible_loss = impossible_loss
                hidden_probs = torch.sigmoid(card_logits)
                impossible_mass = hidden_probs[impossible_mask].mean()
            else:
                impossible_loss = torch.tensor(0.0, device=logits.device)

            hidden_loss = pos_loss + impossible_penalty_weight * impossible_loss
        
        loss = (
            policy_loss
            + 0.5 * value_loss
            + entropy_loss
            + pts_loss
            + hidden_loss_weight * hidden_loss
        )
        
        # bid_value is at index 32, normalized as (val - 120)/300.0
        # point_pred my_pts is normalized as pts / 420.0
        # If the bid is higher than predicted score, add a heavy penalty
        if is_bid.any():
            bid_values_norm = chosen_feats[is_bid, 32]
            # convert bid norm back to actual bid, then norm for 420
            bid_actual = bid_values_norm * 300.0 + 120.0
            
            # Predict our team's points (assuming pov=0, so index 0)
            predicted_pts_actual = pts_pred[is_bid, 0] * 420.0
            
            # (Removed the exponential overbid padding because it artificially exploded the sigmoid bounds)

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"\n[DEBUG] NaN detected. policy: {policy_loss.item()}, value: {value_loss.item()}, entropy: {entropy_loss.item()}, pts_loss: {pts_loss.item()}")
        
        # Skip this batch entirely to prevent corrupting gradients
        return {
            "total": float('nan'), 
            "policy": float('nan'), 
            "value": float('nan'),
            "entropy": float('nan'),
            "pts": float('nan'),
            "hidden": float('nan'),
            "hidden_pos_loss": float('nan'),
            "hidden_impossible_loss": float('nan'),
            "hidden_pos_acc": float('nan'),
            "impossible_mass": float('nan'),
            "approx_kl": float('nan'),
            "clipfrac": float('nan'),
            "forced_imitation": float('nan'),
        }

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    
    # Check for NaN/Inf in gradients after unscaling, before clipping
    grads_valid = True
    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                grads_valid = False
                break
                
    if grads_valid:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        
    scaler.update()
    
    return {
        "total": loss.item() if grads_valid else float('nan'), 
        "policy": policy_loss.item() if grads_valid else float('nan'), 
        "value": value_loss.item() if grads_valid else float('nan'),
        "entropy": entropy.item() if grads_valid else float('nan'),
        "pts": pts_loss.item() if grads_valid else float('nan'),
        "hidden": hidden_loss.item() if grads_valid else float('nan'),
        "hidden_pos_loss": hidden_pos_loss.item() if grads_valid else float('nan'),
        "hidden_impossible_loss": hidden_impossible_loss.item() if grads_valid else float('nan'),
        "hidden_pos_acc": hidden_pos_acc.item() if grads_valid else float('nan'),
        "impossible_mass": impossible_mass.item() if grads_valid else float('nan'),
        "approx_kl": approx_kl.item() if grads_valid else float('nan'),
        "clipfrac": clipfrac.item() if grads_valid else float('nan'),
        "forced_imitation": forced_imitation_loss.item() if grads_valid else float('nan'),
    }
