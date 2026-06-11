#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
ppo_better_train.py  —  Improved PPO-only trainer. Additive; does not modify the base codebase.

Fixes the two reasons the base PPO-only baseline stalled on SimpleDoorKey:
  (1) single non-vectorized env  -> 8 PARALLEL envs (a sync collector)
  (2) Monte-Carlo advantage      -> GAE(lambda) (gae.compute_gae)

Reuses the base codebase's pieces WITHOUT editing them:
  - ActorCritic model : algos.MLPBase (algos/model.py)
  - SimpleDoorKey env : utils.make_env_fn + env registration (env/__init__.py)
  - obs preprocessing : utils.get_obss_preprocessor

PPO-only = NO teacher. GPU (cuda), seed 42.

Run:  set PYTHONUTF8=1  &&  .venv\\Scripts\\python.exe ppo_better_train.py [--iters N] [--smoke]
"""
import sys, os, csv, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn

import env          # noqa: F401  (registers MiniGrid-SimpleDoorKey-* on import)
import register_doorkey8x8  # noqa: F401  (additively registers MiniGrid-SimpleDoorKey-Min8-Max8-View3)
import utils
import algos
from gae import compute_gae

# ── config ────────────────────────────────────────────────────────────────────
ENV_KEY        = "MiniGrid-SimpleDoorKey-Min5-Max10-View3"
NUM_ENVS       = 8
ROLLOUT_T      = 160          # 8 * 160 = 1280 steps/iter (~ baseline's ~1500)
TOTAL_ITERS    = 1000         # fair long shot (~1.3M steps); paper says pure RL needs millions
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
VALUE_COEF     = 0.5
ENTROPY_COEF   = 0.05         # START high to keep exploring (sparse long-horizon task)
ENTROPY_FINAL  = 0.01         # ANNEAL linearly to this by the final iteration
PPO_EPOCHS     = 4
NUM_MINIBATCH  = 4
LR             = 2.5e-4
ADAM_EPS       = 1e-5
MAX_GRAD_NORM  = 0.5
SEED           = 42
EVAL_INTERVAL  = 10
NUM_EVAL       = 10
# Relaxed early-stop: only bail if TRULY dead-flat well into training.
GUARD_MIN_ITER = 400          # never stop before this
GUARD_WINDOW   = 50           # rolling window
GUARD_SR       = 0.02         # best 50-iter train success must exceed this past iter 400

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "better_ppo")
DEVICE = "cuda"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_t(obs_np):
    return torch.as_tensor(np.asarray(obs_np), dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def evaluate(model, eval_env, n_episodes, deterministic=False):
    """No-teacher eval: n episodes, success = return > 0 (door opened)."""
    succ, rets, lens = [], [], []
    for _ in range(n_episodes):
        o, _ = eval_env.reset()
        img = o["image"]
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            dist, _, _ = model(to_t(img[None]))
            a = int(torch.argmax(dist.probs, dim=-1)) if deterministic else int(dist.sample())
            o, r, term, trunc, _ = eval_env.step(a)
            img = o["image"]
            ep_ret += float(r)
            ep_len += 1
            done = term or trunc
        succ.append(1.0 if ep_ret > 0 else 0.0)
        rets.append(ep_ret)
        lens.append(ep_len)
    return float(np.mean(succ)), float(np.mean(rets)), float(np.mean(lens))


def main():
    global ENV_KEY, ROLLOUT_T, RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=TOTAL_ITERS)
    ap.add_argument("--smoke", action="store_true", help="3 iters, no early stop, no file write")
    ap.add_argument("--no-guard", dest="no_guard", action="store_true",
                    help="disable the relaxed early-stop guard; run all iters to completion")
    ap.add_argument("--env-key", dest="env_key", default=ENV_KEY,
                    help="gym env id (default: SimpleDoorKey)")
    ap.add_argument("--rollout", type=int, default=ROLLOUT_T,
                    help="per-env rollout horizon (8*rollout = steps/iter)")
    ap.add_argument("--results-subdir", dest="results_subdir", default="better_ppo",
                    help="subfolder under results/ for metrics/model/plots")
    args = ap.parse_args()
    total_iters = 3 if args.smoke else args.iters

    # additive env/rollout/results overrides (SimpleDoorKey defaults preserved)
    ENV_KEY     = args.env_key
    ROLLOUT_T   = args.rollout
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", args.results_subdir)

    assert torch.cuda.is_available(), "CUDA not available — aborting GPU run"
    set_seed(SEED)
    print(f"[better-ppo] GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print(f"[better-ppo] {NUM_ENVS} envs x {ROLLOUT_T} steps = {NUM_ENVS*ROLLOUT_T} steps/iter | "
          f"iters={total_iters} | GAE(g={GAMMA},lam={GAE_LAMBDA}) | "
          f"ent={ENTROPY_COEF}->{ENTROPY_FINAL} (anneal) | lr={LR}")

    # ── build envs (reuse base env builder) ───────────────────────────────────
    envs = [utils.make_env_fn(ENV_KEY)() for _ in range(NUM_ENVS)]
    eval_env = utils.make_env_fn(ENV_KEY)()
    obs_space, _ = utils.get_obss_preprocessor(envs[0].observation_space)
    action_dim = envs[0].action_space.n
    img_shape = obs_space["image"]          # (10,10,4)
    print(f"[better-ppo] obs_space={obs_space} action_dim={action_dim}")

    # ── model (reuse base ActorCritic) + optimizer ────────────────────────
    model = algos.MLPBase(obs_space, action_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, eps=ADAM_EPS)
    assert next(model.parameters()).is_cuda

    # ── init env states (seed each env distinctly, once) ──────────────────────
    cur_obs = np.zeros((NUM_ENVS, *img_shape), dtype=np.float32)
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=SEED + i)
        cur_obs[i] = o["image"]
    ep_ret = np.zeros(NUM_ENVS, dtype=np.float64)
    ep_len = np.zeros(NUM_ENVS, dtype=np.int64)

    # storage
    S = (ROLLOUT_T, NUM_ENVS)
    history = dict(iter=[], success_rate=[], eval_success_rate=[], average_return=[],
                   episode_length=[], entropy=[], eval_return=[], eval_ep_len=[], total_steps=[])
    total_steps = 0
    last_eval = dict(sr=np.nan, ret=np.nan, length=np.nan)
    t_start = time.time()

    for it in range(total_iters):
        # linear entropy-coef anneal: ENTROPY_COEF -> ENTROPY_FINAL over training
        frac = it / max(total_iters - 1, 1)
        ent_coef_now = ENTROPY_COEF + (ENTROPY_FINAL - ENTROPY_COEF) * frac

        obs_buf   = np.zeros((*S, *img_shape), dtype=np.float32)
        act_buf   = np.zeros(S, dtype=np.int64)
        logp_buf  = np.zeros(S, dtype=np.float32)
        val_buf   = np.zeros(S, dtype=np.float32)
        rew_buf   = np.zeros(S, dtype=np.float32)
        term_buf  = np.zeros(S, dtype=np.float32)
        trunc_buf = np.zeros(S, dtype=np.float32)
        tval_buf  = np.zeros(S, dtype=np.float32)   # V(final_obs) at truncated steps

        ep_returns, ep_lengths, ep_success = [], [], []

        # ── collect rollout ───────────────────────────────────────────────────
        for t in range(ROLLOUT_T):
            with torch.no_grad():
                dist, value, _ = model(to_t(cur_obs))
                action = dist.sample()
                logp = dist.log_prob(action)
            a_np = action.cpu().numpy()
            obs_buf[t] = cur_obs
            act_buf[t] = a_np
            logp_buf[t] = logp.cpu().numpy()
            val_buf[t] = value.cpu().numpy()

            for i, e in enumerate(envs):
                o, r, term, trunc, _ = e.step(int(a_np[i]))
                rew_buf[t, i] = r
                term_buf[t, i] = float(term)
                trunc_buf[t, i] = float(trunc)
                ep_ret[i] += r
                ep_len[i] += 1
                if term or trunc:
                    ep_returns.append(ep_ret[i])
                    ep_lengths.append(int(ep_len[i]))
                    ep_success.append(1.0 if ep_ret[i] > 0 else 0.0)
                    if trunc and not term:    # bootstrap value of the true final obs
                        with torch.no_grad():
                            tv = model(to_t(o["image"][None]))[1]
                        tval_buf[t, i] = float(tv)
                    o2, _ = e.reset()
                    cur_obs[i] = o2["image"]
                    ep_ret[i] = 0.0
                    ep_len[i] = 0
                else:
                    cur_obs[i] = o["image"]

        total_steps += ROLLOUT_T * NUM_ENVS

        # value of the state each env is currently in (for non-done last steps)
        with torch.no_grad():
            last_values = model(to_t(cur_obs))[1]

        # ── GAE ───────────────────────────────────────────────────────────────
        rew_t  = torch.as_tensor(rew_buf, device=DEVICE)
        val_t  = torch.as_tensor(val_buf, device=DEVICE)
        term_t = torch.as_tensor(term_buf, device=DEVICE)
        trunc_t = torch.as_tensor(trunc_buf, device=DEVICE)
        tval_t = torch.as_tensor(tval_buf, device=DEVICE)
        adv_t, ret_t = compute_gae(rew_t, val_t, term_t, trunc_t, last_values, tval_t,
                                   gamma=GAMMA, lam=GAE_LAMBDA)

        # ── flatten for minibatch PPO ─────────────────────────────────────────
        B = ROLLOUT_T * NUM_ENVS
        b_obs  = to_t(obs_buf.reshape(B, *img_shape))
        b_act  = torch.as_tensor(act_buf.reshape(B), device=DEVICE, dtype=torch.long)
        b_logp = torch.as_tensor(logp_buf.reshape(B), device=DEVICE)
        b_val  = val_t.reshape(B)
        b_adv  = adv_t.reshape(B)
        b_ret  = ret_t.reshape(B)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        mb_size = B // NUM_MINIBATCH
        ent_log = []
        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(B, device=DEVICE)
            for s in range(0, B, mb_size):
                mb = idx[s:s + mb_size]
                dist, value, _ = model(b_obs[mb])
                new_logp = dist.log_prob(b_act[mb])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logp - b_logp[mb])
                a_mb = b_adv[mb]
                pg1 = -a_mb * ratio
                pg2 = -a_mb * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                policy_loss = torch.max(pg1, pg2).mean()
                # clipped value loss
                v_clipped = b_val[mb] + torch.clamp(value - b_val[mb], -CLIP_EPS, CLIP_EPS)
                vl1 = (value - b_ret[mb]).pow(2)
                vl2 = (v_clipped - b_ret[mb]).pow(2)
                value_loss = 0.5 * torch.max(vl1, vl2).mean()
                loss = policy_loss + VALUE_COEF * value_loss - ent_coef_now * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()
                ent_log.append(float(entropy))

        # ── metrics ───────────────────────────────────────────────────────────
        tr_sr   = float(np.mean(ep_success)) if ep_success else (history["success_rate"][-1] if history["success_rate"] else 0.0)
        tr_ret  = float(np.mean(ep_returns)) if ep_returns else 0.0
        tr_len  = float(np.mean(ep_lengths)) if ep_lengths else 150.0
        ent_m   = float(np.mean(ent_log))

        if it % EVAL_INTERVAL == 0 or it == total_iters - 1:
            es, er, el = evaluate(model, eval_env, NUM_EVAL, deterministic=False)
            last_eval = dict(sr=es, ret=er, length=el)

        history["iter"].append(it)
        history["success_rate"].append(tr_sr)
        history["eval_success_rate"].append(last_eval["sr"])
        history["average_return"].append(tr_ret)
        history["episode_length"].append(tr_len)
        history["entropy"].append(ent_m)
        history["eval_return"].append(last_eval["ret"])
        history["eval_ep_len"].append(last_eval["length"])
        history["total_steps"].append(total_steps)

        el = time.time() - t_start
        flag = "OK" if tr_sr > 0 else ".."
        print(f"[better-ppo] [{it:3d}/{total_iters}] train_sr={tr_sr:.3f} eval_sr={last_eval['sr']:.3f} "
              f"ret={tr_ret:.3f} len={tr_len:5.1f} ent={ent_m:.3f} entc={ent_coef_now:.3f} "
              f"eps={len(ep_success):2d} steps={total_steps:,} {flag} {el:.0f}s", flush=True)

        # ── relaxed early-stop: only bail if TRULY dead-flat past GUARD_MIN_ITER ──
        if (not args.smoke) and (not args.no_guard) and it >= GUARD_MIN_ITER:
            sr = history["success_rate"]
            best_roll = max(np.mean(sr[i:i + GUARD_WINDOW])
                            for i in range(0, max(1, len(sr) - GUARD_WINDOW + 1)))
            if best_roll < GUARD_SR:
                print(f"\n[better-ppo] EARLY STOP at iter {it}: best {GUARD_WINDOW}-iter train success "
                      f"{best_roll:.3f} < {GUARD_SR} past iter {GUARD_MIN_ITER} — bare PPO not learning. "
                      f"Stopping.", flush=True)
                break

    # ── save ──────────────────────────────────────────────────────────────────
    if not args.smoke:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        cols = list(history.keys())
        with open(os.path.join(RESULTS_DIR, "metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in range(len(history["iter"])):
                w.writerow([history[c][r] for c in cols])
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))
        print(f"[better-ppo] saved metrics + model -> {RESULTS_DIR}")

    sr = history["success_rate"]
    es = [x for x in history["eval_success_rate"] if x == x]
    best_tr = max(sr) if sr else 0.0
    # crude breakthrough: first iter where 5-iter rolling train success >= 0.5
    bt = None
    for i in range(len(sr)):
        if len(sr[i:i + 5]) == 5 and np.mean(sr[i:i + 5]) >= 0.5:
            bt = history["iter"][i]
            break
    print("\n" + "=" * 60)
    print(f"[better-ppo] DONE in {(time.time()-t_start)/60:.1f} min | model on {next(model.parameters()).device}")
    print(f"[better-ppo] final train_sr={sr[-1]:.3f} | best train_sr={best_tr:.3f} | "
          f"final eval_sr={history['eval_success_rate'][-1]:.3f} | best eval_sr={max(es) if es else float('nan'):.3f}")
    print(f"[better-ppo] entropy first->last: {history['entropy'][0]:.3f} -> {history['entropy'][-1]:.3f}")
    print(f"[better-ppo] breakthrough (5-iter train SR>=0.5): itr {bt}")
    print("[better-ppo] DONE_OK")


if __name__ == "__main__":
    main()
