# PPO-only Baselines on MiniGrid DoorKey Environments

Bare Proximal Policy Optimization (PPO) baselines on five sparse-reward MiniGrid
DoorKey environments, trained with environment reward only and **no teacher and no
reward shaping**. These runs are the no-teacher reference point for an LLM-teacher
comparison study: they measure how far a standard deep-RL agent gets on each task on
its own, so that teacher-guided methods can be compared against an honest floor. This
repository is **only the PPO-only baseline track** — the teacher / reflection work is
developed and evaluated separately and is not included here.

## Method

- **Algorithm:** PPO with **8 parallel environments** and **Generalized Advantage
  Estimation** (GAE), gamma = 0.99, lambda = 0.95.
- **Reward:** environment reward only — **no teacher, no intrinsic/curiosity reward, no
  shaping**.
- **Other settings:** clip 0.2, value-loss coef 0.5, entropy coef annealed 0.05 -> 0.01,
  Adam lr 2.5e-4, 4 PPO epochs x 4 minibatches per update, gradient-norm clip 0.5,
  **seed 42**, run on **GPU (CUDA)**.
- **Rollout / episodes:** per-env horizon 188 steps -> 8 x 188 = 1504 env steps per
  iteration; with an episode cap of 150 steps this is about **10 episodes per iteration**.
- **Additive by design:** the agent reuses the base codebase's ActorCritic model
  (`algos`) and the environments (`env`) **unchanged**. All baseline code is additive —
  the shared scripts live at the repo root and each environment's outputs live in its own
  section folder. The base codebase is not modified.

## Environments

| Environment | gym id | Definition |
|---|---|---|
| SimpleDoorKey | `MiniGrid-SimpleDoorKey-Min5-Max10-View3` | room size 5–10, agent view 3, max_steps 150 |
| LavaDoorKey | `MiniGrid-LavaDoorKey-Min5-Max10-View3` | room size 5–10, agent view 3, max_steps 150 |
| ColoredDoorKey | `MiniGrid-ColoredDoorKey-Min5-Max10-View3` | room size 5–10, 2 keys, agent view 3, max_steps 150 |
| 8x8 DoorKey | `MiniGrid-SimpleDoorKey-Min8-Max8-View3` | fixed 8x8 room (`DoorKeyEnv` with `minRoomSize=maxRoomSize=8`, `agent_view_size=3`, `max_steps=150`), registered additively by `register_doorkey8x8.py` |
| Larger DoorKey (12-14) | `MiniGrid-SimpleDoorKey-Min12-Max14-View3` | room size 12–14, agent view 3, max_steps 196 (`DoorKeyEnv` with `minRoomSize=12, maxRoomSize=14, agent_view_size=3, max_steps=196`), registered additively by `register_doorkey12_14.py` |

The 8x8 and 12-14 environments reuse the base `DoorKeyEnv` class with only the room-size
(and, for 12-14, the `max_steps`) kwargs changed, so they stay inside the same observation
family as the other tasks. SimpleDoorKey, LavaDoorKey and ColoredDoorKey were run for **200
iterations** (~10 episodes/iter); SimpleDoorKey and the larger 12-14 grid additionally have
full **1000-iteration** records (the data in `ppo_simpledoorkey/` and `ppo_doorkey12_14/`).
On the 12-14 grid the episode cap is 196 steps, so a longer rollout horizon (245) keeps the
~10 episodes/iter budget.

## Results

Success = the agent opens the door (episode return > 0). Values are read directly from each
environment's `metrics.csv`.

| Environment | Iterations | Env steps | Train success (final / best) | Eval success (final / best) |
|---|---|---|---|---|
| SimpleDoorKey | 1000 | 1,280,000 | 0.000 / 0.222 | 0.000 / 0.100 |
| LavaDoorKey | 200 | 300,800 | 0.000 / 0.111 | 0.000 / 0.100 |
| ColoredDoorKey | 200 | 300,800 | 0.000 / 0.125 | 0.000 / 0.100 |
| 8x8 DoorKey | 200 | 300,800 | 0.000 / 0.125 | 0.000 / 0.000 |
| Larger DoorKey (12-14) | 1000 | 1,960,000 | 0.000 / 0.111 | 0.000 / 0.000 |

**Finding.** Bare PPO-only stays at essentially **0% success** on all five sparse-reward
DoorKey tasks, even with GAE and 8 parallel environments and up to ~1.3M environment steps
on SimpleDoorKey. The only non-zero success rates are isolated single-iteration flukes (the
"best" column), never a sustained policy, and final success is 0.000 everywhere. This is the
**intended weak baseline**: it isolates the difficulty of sparse-reward exploration on these
tasks and provides the reference floor that the teacher-guided methods are meant to improve
on. The larger 12-14 grid — harder than the 5–10 tasks, with a bigger observation and a
196-step cap — likewise stays flat at 0.000 final success across a full 1000-iteration
(~1.96M-step) run, confirming the floor holds as the task scales up. See each folder's
`*_success.png` and `*_success_vs_steps.png` for the learning curves.

## Repository layout

Shared code at the repo root, one self-contained folder per environment:

```
ppo_better_train.py        # PPO + GAE trainer (8 parallel envs); env-parametric via flags
gae.py                     # GAE(lambda) advantage estimation
plot_better_ppo.py         # learning-curve plotting (6-figure suite, 200 dpi)
register_doorkey8x8.py     # additively registers the 8x8 DoorKey gym id
register_doorkey12_14.py   # additively registers the larger Min12-Max14 DoorKey gym id
env/  algos/  utils/       # base environments + ActorCritic + helpers (dependencies)
requirements.txt

ppo_simpledoorkey/   ppo_lavadoorkey/   ppo_coloreddoorkey/   ppo_doorkey8x8/   ppo_doorkey12_14/
```

Each `ppo_<env>/` folder contains:

- `better_ppo_success.png` — train + eval success rate vs iteration (with a secondary env-steps axis)
- `better_ppo_success_vs_steps.png` — success rate vs total environment steps
- `better_ppo_return.png` — average episode return
- `better_ppo_eplen.png` — episode length
- `better_ppo_entropy.png` — policy entropy
- `better_ppo_combined.png` — 2x2 panel of the above
- `metrics.csv` — per-iteration metrics (the source of the results table)
- `history.json` — the same metrics in JSON
- `model.pt` — the final trained model weights

## Reproduce

Dependencies: `env/`, `algos/`, `utils/`, and the packages in `requirements.txt`. A
**GPU (CUDA)** is expected. On Windows, set UTF-8 first.

```bat
set PYTHONUTF8=1
```

SimpleDoorKey (1000 iterations):
```bat
python ppo_better_train.py --env-key MiniGrid-SimpleDoorKey-Min5-Max10-View3 --rollout 160 --iters 1000 --no-guard --results-subdir ppo_simpledoorkey
python plot_better_ppo.py --subdir ppo_simpledoorkey --title "Bare PPO-only (8 envs + GAE) - SimpleDoorKey, seed 42"
```

LavaDoorKey:
```bat
python ppo_better_train.py --env-key MiniGrid-LavaDoorKey-Min5-Max10-View3 --rollout 188 --iters 200 --no-guard --results-subdir ppo_lavadoorkey
python plot_better_ppo.py --subdir ppo_lavadoorkey --title "Bare PPO-only (8 envs + GAE) - LavaDoorKey, seed 42"
```

ColoredDoorKey:
```bat
python ppo_better_train.py --env-key MiniGrid-ColoredDoorKey-Min5-Max10-View3 --rollout 188 --iters 200 --no-guard --results-subdir ppo_coloreddoorkey
python plot_better_ppo.py --subdir ppo_coloreddoorkey --title "Bare PPO-only (8 envs + GAE) - ColoredDoorKey, seed 42"
```

8x8 DoorKey (registered by `register_doorkey8x8.py`, imported automatically by the trainer):
```bat
python ppo_better_train.py --env-key MiniGrid-SimpleDoorKey-Min8-Max8-View3 --rollout 188 --iters 200 --no-guard --results-subdir ppo_doorkey8x8
python plot_better_ppo.py --subdir ppo_doorkey8x8 --title "Bare PPO-only (8 envs + GAE) - DoorKey 8x8, seed 42"
```

Larger DoorKey 12-14 (1000 iterations; registered by `register_doorkey12_14.py`, imported automatically by the trainer). The 196-step episode cap uses a longer rollout horizon (245) to keep ~10 episodes/iter (8 x 245 = 1960 ≈ 10 x 196):
```bat
python ppo_better_train.py --env-key MiniGrid-SimpleDoorKey-Min12-Max14-View3 --rollout 245 --iters 1000 --no-guard --results-subdir ppo_doorkey12_14
python plot_better_ppo.py --subdir ppo_doorkey12_14 --title "Bare PPO-only (8 envs + GAE) - SimpleDoorKey Min12-Max14, seed 42"
```

Each run writes `metrics.csv`, `history.json`, `model.pt`, and the six plots into
`results/<results-subdir>/`. The results committed in this repository were then reorganized
into the top-level `ppo_<env>/` section folders — so a re-run reproduces the same data, it
just lands under `results/` first.
