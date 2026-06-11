#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
gae.py  —  Generalized Advantage Estimation. Additive; does not modify the base codebase.

The base codebase's algos/buffer.py uses Monte-Carlo advantage (returns - values) and
leaves `lam` unused. This is a clean, correct GAE(lambda) that distinguishes
termination (no bootstrap) from truncation/timeout (bootstrap with V(final_obs)).

compute_gae(rewards, values, terminated, truncated, last_values, trunc_values,
            gamma=0.99, lam=0.95) -> (advantages, returns), all torch [T, N].
"""
import torch


def compute_gae(rewards, values, terminated, truncated, last_values,
                trunc_values, gamma: float = 0.99, lam: float = 0.95):
    """
    Parameters (all torch tensors on the same device):
      rewards      [T, N]  reward received after acting in s_t
      values       [T, N]  V(s_t) from the critic at collection time
      terminated   [T, N]  bool/float — episode ended at goal (no future value)
      truncated    [T, N]  bool/float — episode cut by time limit (bootstrap)
      last_values  [N]      V(s_T): value of the state each env is in AFTER the
                            final rollout step (used only for non-done last steps)
      trunc_values [T, N]  V(final_obs) at truncated steps, 0 elsewhere
    Returns:
      advantages   [T, N]
      returns      [T, N]  = advantages + values  (critic targets)
    """
    T, N = rewards.shape
    terminated = terminated.float()
    truncated = truncated.float()
    done = torch.clamp(terminated + truncated, max=1.0)  # episode boundary

    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        next_val = last_values if t == T - 1 else values[t + 1]
        # bootstrap value of the state reached after step t:
        #   terminated -> 0 ; truncated -> V(final_obs) ; otherwise -> V(s_{t+1})
        nv = next_val * (1.0 - done[t]) + trunc_values[t] * truncated[t]
        delta = rewards[t] + gamma * nv - values[t]
        last_adv = delta + gamma * lam * (1.0 - done[t]) * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


# ── self-check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    # Case 1: single env, 3 steps, no done — compare against manual GAE.
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    term = torch.zeros(3, 1)
    trunc = torch.zeros(3, 1)
    last_v = torch.tensor([0.1])
    trunc_v = torch.zeros(3, 1)
    g, lam = 0.99, 0.95
    adv, ret = compute_gae(rewards, values, term, trunc, last_v, trunc_v, g, lam)

    # manual
    d2 = rewards[2, 0] + g * last_v[0] - values[2, 0]
    a2 = d2
    d1 = rewards[1, 0] + g * values[2, 0] - values[1, 0]
    a1 = d1 + g * lam * a2
    d0 = rewards[0, 0] + g * values[1, 0] - values[0, 0]
    a0 = d0 + g * lam * a1
    expected = torch.tensor([a0, a1, a2])
    assert torch.allclose(adv[:, 0], expected, atol=1e-6), (adv[:, 0], expected)
    print("[gae] case1 (no done) OK:", adv[:, 0].tolist())

    # Case 2: terminated at last step -> no bootstrap from last_values.
    term2 = torch.tensor([[0.0], [0.0], [1.0]])
    adv2, _ = compute_gae(rewards, values, term2, trunc, last_v, trunc_v, g, lam)
    d2b = rewards[2, 0] + g * 0.0 - values[2, 0]   # terminated => nv = 0
    assert torch.allclose(adv2[2, 0], d2b, atol=1e-6)
    print("[gae] case2 (terminated) OK: adv[-1] =", float(adv2[2, 0]))

    # Case 3: truncated at last step -> bootstrap with trunc_values.
    trunc3 = torch.tensor([[0.0], [0.0], [1.0]])
    trunc_v3 = torch.tensor([[0.0], [0.0], [0.7]])
    adv3, _ = compute_gae(rewards, values, term, trunc3, last_v, trunc_v3, g, lam)
    d2c = rewards[2, 0] + g * 0.7 - values[2, 0]   # truncated => nv = V(final)=0.7
    assert torch.allclose(adv3[2, 0], d2c, atol=1e-6)
    print("[gae] case3 (truncated) OK: adv[-1] =", float(adv3[2, 0]))

    print("[gae] ALL SELF-CHECKS PASSED")
