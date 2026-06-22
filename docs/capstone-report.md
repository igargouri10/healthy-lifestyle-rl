# Healthy Lifestyle RL — Capstone Report (V3)

**Author:** Ismail Gargouri
**Course:** EECS 590, Reinforcement Learning, Spring 2026
**Repository:** `healthy-lifestyle-rl`

## Abstract

This capstone investigates whether a single reinforcement-learning agent, trained on a custom Gymnasium-compatible simulator, can outperform two non-learning baselines (uniform random and a fitness-planner heuristic) at the task of recommending small daily adjustments to a simulated user's calorie target, sleep target, step target, and protein target across a 60-day horizon. The agent is constrained by a configurable safety layer that clips unsafe action proposals before they reach the transition model, and is trained against a reward decomposition that explicitly penalizes unsafe proposals, rapid daily changes, and weight-trajectory instability. Two algorithms are implemented: a custom-PyTorch DQN agent (chosen because the course derived Q-learning in detail) and a Stable-Baselines3 PPO agent (chosen so the project has a stable, well-tested on-policy reference implementation). Bayesian (Optuna TPE) hyperparameter search is applied to the DQN agent. All four policies are evaluated across five simulated profiles. The project also documents the methods that were deliberately *not* implemented (multi-agent, swarm, robotics simulators, offline RL, Sim2Real, meta-RL, hierarchical RL) and justifies each rejection in `docs/decisions.md`.

## 1. Introduction

A user trying to maintain a healthy lifestyle makes dozens of small decisions every day: how much to eat, when to sleep, when to push activity, when to rest. Most of these choices are fine in isolation but couple in ways that are hard to see ahead of time — pushing activity hard on a low-sleep day degrades recovery, which degrades next-day adherence, which degrades next-week consistency. The effects of today's choices show up days later in the recovery and weight-trend signals. This temporal structure is exactly what makes the problem interesting from an RL perspective: it is a sequential decision problem with delayed reward [1], and exactly the kind of problem that a competent rule-based planner can approximate but only an agent with credit assignment can learn end-to-end.

The goal of this capstone is not to *deploy* such an agent — the safety and validation requirements for real-world health recommendation are well outside what belongs in a semester project — but to build the simulator, the safety layer, the agents, and the evaluation harness that would be the starting point for a more serious effort.

## 2. Problem formulation

We model one day in the user's life as one step in a Markov decision process [1]. The agent observes a normalized 18-dimensional summary of the user's static profile and dynamic state, picks one of 8 discrete "adjustment" actions, the safety layer translates that action into per-day target deltas and clips them, and the transition model rolls forward one simulated day. The reward is a weighted sum of eight components.

The single-agent framing is deliberate and is discussed at length in `docs/decisions.md`. The unit of optimization is one user, not a group or a fleet, which rules out multi-agent and coordination methods.

## 3. MDP design

| Component | Definition |
|---|---|
| **States** | 18-dim normalized vector covering age, sex, height, initial weight, goal-type code, current weight, 7-day weight trend, calorie target, actual calorie intake, protein/carbs/fat (g), sleep hours, steps, recovery score, adherence score, energy score, and normalized day index. Represented as `gymnasium.spaces.Box(low=-2.0, high=2.0, shape=(18,))`. |
| **Actions** | 8 discrete adjustments: `{maintain, decrease cal, increase cal, increase activity, prioritize recovery, increase sleep, raise protein, rebalance carbs}`. The agent never sets absolute values — only adjustments, which the safety layer clips. `gymnasium.spaces.Discrete(8)`. |
| **Transition** | Stochastic, implemented in `src/lifestyle_rl/simulation/transition_model.py`. Adherence-weighted realization of targets, recovery driven by sleep and load, slow weight evolution from the 7700 kcal/kg energy-balance approximation. |
| **Reward** | Decomposed into 8 components (see §5). |
| **Horizon** | 60 days by default; configurable to 30 (quick training) or 90 (final eval). |
| **Discount** | `γ = 0.99`. |

The formal MDP notation `(S, A, P, R, γ)` [1] applies directly: the state space S is the 18-dim Box, the action space A is Discrete(8), P is the stochastic transition model, R is the decomposed reward function, and γ = 0.99.

## 4. Environment design

`HealthyLifestyleEnv` implements the standard `reset → step` Gymnasium API [5] and a text `render`. Each `step()` returns the standard 5-tuple `(obs, reward, terminated, truncated, info)` plus an `info` dictionary that exposes the raw action, safe action, whether a safety intervention was applied, the clipped components, the action's natural-language meaning, current weight, weight trend, adherence/recovery/energy scores, calorie target, sleep hours, steps, and the per-component reward breakdown. Having all of this in `info` made debugging far easier than I expected — when something looked wrong in the training curve, I could add a single `print(info["reward_components"])` to see exactly which component was misbehaving.

Profiles are drawn from a library of five personas (`maintenance`, `general_wellness`, `activity_consistency`, `gradual_management`, `performance_support`) and optionally jittered with Gaussian noise on age, height, and weight. The jitter is reproducible — it is driven by the env's seeded RNG — so rollouts are repeatable.

## 5. Reward decomposition

The reward at each step combines positive and negative components:

```
r =   w_goal     · goal_progress(state, profile)
    + w_adh      · adherence_score
    + w_sleep    · bell(sleep_hours, 7.0, 8.5)
    + w_act      · bell(steps, profile_baseline ± window)
    + w_rec      · recovery_score
    - w_unsafe   · unsafe_proposal
    - w_rapid    · rapid_change_magnitude
    - w_stab     · per_day_weight_swing
```

`bell(x, lo, hi)` returns 1.0 inside the target window and falls off as a Gaussian outside. The explicit penalty terms (unsafe, rapid change, instability) were important to get right early — the first version had `+unsafe_action_penalty * unsafe` which *rewarded* the agent for triggering safety interventions. That was caught by the test suite, not by inspection; it's the main reason `test_rewards.py::test_unsafe_action_gets_penalty` exists.

Goal-progress targets per goal type live in `GOAL_TARGETS` in `reward_functions.py`; weights live in `configs/env.yaml` and are configurable.

## 6. Algorithms

### 6.1 Random and rule-based baselines

The random policy samples uniformly from the action space. The rule-based policy is a deterministic heuristic: if adherence is low, maintain; if recovery is low, recover; if sleep is low, sleep; if the 7-day weight trend is outside the goal window, nudge calories. Both are in `src/lifestyle_rl/agents/baselines.py`. These baselines are important because they set the floor the RL agents need to beat. As it turns out, the rule-based heuristic is a reasonably strong competitor on the synthetic environment — which is actually a useful calibration, not a problem.

### 6.2 DQN (custom PyTorch)

DQN [2, 3] was the first algorithm I implemented because the course derived the Q-learning update rule in detail and I wanted the implementation to mirror that derivation directly. The core ideas from the Nature paper [2] are all present:

- **Experience replay** [3]: a circular buffer of 50k transitions. Each training step samples a minibatch uniformly at random, breaking the correlation between consecutive transitions that would destabilize gradient descent.
- **Target network**: a second copy of the Q-network whose weights are held fixed for `target_update_freq` environment steps, then hard-synced. This stabilizes the Bellman target `r + γ·max_a' Q_target(s', a')` during training [2].
- **Epsilon-greedy exploration**: ε decays linearly from 1.0 to 0.05 over `epsilon_decay_episodes` episodes [1, 3].
- **Huber loss**: smooth L1 loss between the predicted Q-value and the Bellman target, which is more robust to large TD errors than MSE [2].
- **Gradient clipping**: `nn.utils.clip_grad_norm_` to `grad_clip=10.0`, which prevents the occasional large gradient from destabilizing the network.

The Q-network is a two-hidden-layer MLP (128 → 128 → 8). With only 18 input features and 8 outputs this is larger than necessary, but it avoids capacity being a bottleneck during Bayesian tuning.

### 6.3 PPO (Stable-Baselines3)

PPO [4] uses a clipped surrogate objective to limit how far a policy update can move from the current policy, which makes it much more stable than vanilla policy gradient. I chose to use Stable-Baselines3 [6] rather than implement PPO from scratch because PPO has several correctness-sensitive components (clipped surrogate, GAE [4b], value-function loss coefficient, entropy bonus) that are easy to get subtly wrong, and the course did not require me to re-implement it. Writing one carefully-derived algorithm from scratch (DQN) and one well-tested library wrapper (PPO) gives a more honest project than two half-correct custom rolls.

The SB3 PPO is configured with `n_steps=512`, `ent_coef=0.01`, and `clip_range=0.2` (the default from the original paper [4]).

### 6.4 Bayesian hyperparameter tuning (Optuna)

`scripts/tune_bayesian.py` uses Optuna's TPE (Tree-structured Parzen Estimator) sampler [7] to search over `{learning_rate, gamma, batch_size, hidden_size, epsilon_decay_episodes}` on the DQN agent. The objective is the mean episode reward over the trailing 20% of the training curve.

The choice of Bayesian search over grid search matters here. With 5 hyperparameters and a plausible grid of 4–5 values each, grid search would be 1024–3125 trials. With episodes that take 10–30 seconds, that budget is prohibitive. TPE concentrates the search budget in regions where the surrogate model predicts the next trial will be most informative [7], which is exactly the efficiency property we need.

## 7. Safety layer

The safety layer is the single most architecturally important piece of the project. It sits *between* the agent and the simulator: the agent emits an action id, the safety layer maps it to per-day deltas and clips those deltas against configurable bounds, and reports the intervention back through `info`. The configurable bounds (in `configs/env.yaml`) are:

| Constraint | Default | What it prevents |
|---|---|---|
| `min_calorie_target` | 1500 kcal | Extreme restriction |
| `max_calorie_target` | 3500 kcal | Extreme surplus |
| `max_daily_calorie_change` | 150 kcal | Large single-day swings |
| `min_sleep_target_h` | 6.5 h | Sleep deprivation |
| `max_sleep_target_h` | 9.5 h | Pathological oversleeping |
| `max_step_increase_per_day` | 1500 steps | Sudden activity spike |
| `min_recovery_threshold` | 0.35 | Training when under-recovered |
| `min_protein_g_per_kg` | 1.2 g/kg | Protein deficiency |
| `max_protein_g_per_kg` | 2.4 g/kg | Excessive protein |

A separate "low-recovery override" forces a recovery action when `recovery_score < min_recovery_threshold`, regardless of what the agent wanted to do. Every intervention is surfaced in the reward as an `unsafe_action_penalty`, so the layer acts as both a hard constraint (clipping) and a soft constraint (penalty) simultaneously — the agent has a learning signal to stay inside the safe envelope voluntarily.

## 8. Experiments

Each policy is evaluated for 20 episodes per profile across all 5 profiles (100 total rollouts per policy). Metrics: mean episode reward, mean safety-intervention rate, mean adherence, mean recovery, and the action-frequency distribution.

The evaluation script saves four plots:

- `results/plots/reward_comparison.png` — mean episode reward by policy
- `results/plots/safety_intervention_comparison.png` — mean interventions/episode by policy
- `results/plots/adherence_comparison.png` — mean adherence score by policy
- `results/plots/action_distribution.png` — per-action frequency, grouped by policy

…plus the underlying per-profile, per-policy table at `results/metrics/evaluation_summary.csv`.

## 9. Results

All four policies were evaluated for 20 episodes per profile across all 5 profiles (100 rollouts per policy, 400 total). Full per-profile numbers are in `results/metrics/evaluation_summary.csv`. Aggregate results:

| Policy | Mean reward (all profiles) | Mean safety interventions / episode |
|---|---|---|
| Random | ≈ 178 | ≈ 3.6 |
| Rule-based | ≈ 207 | ≈ 0.0 |
| DQN (250 episodes) | ≈ 198 | ≈ 0.1 |
| PPO (SB3, ~500 ep) | ≈ 202* | ≈ 6.1* |

*PPO average is skewed by an anomaly on `gradual_management` (see below and `docs/technical-challenges.md`).

Per-profile breakdown:

| Profile | Random | Rule-based | DQN | PPO |
|---|---|---|---|---|
| maintenance | 177 | 206 | **202** | 205 |
| general_wellness | 181 | 208 | 197 | 197 |
| activity_consistency | 182 | 207 | 192 | **208** |
| gradual_management | 175 | 208 | 197 | 200 |
| performance_support | 175 | 208 | **203** | 201 |

**Key findings:**

The rule-based heuristic consistently scores ≈205–208 with zero safety interventions on every profile — the ceiling that the RL agents need to approach. This is not a trivial baseline: the heuristic was designed knowing exactly what the reward components are, while the RL agents have to discover the optimal policy by trial and error from the reward signal alone.

The DQN used for final evaluation was trained for 250 episodes and reaches 192–203 depending on the profile. The training curve (`results/plots/dqn_training_curve.png`) shows clear improvement from ≈184 at episode 10 to ≈197 by episode 200, when exploration ends (`epsilon=0.05`). I separately trained a 500-episode variant to check whether more training would close the gap to the rule-based; it did not — the reward plateaus after episode 200, oscillating in the 192–202 band, and the extra episodes actually degraded safety behavior (see the note at the end of this section). The plateau tells us something important: DQN is not failing to converge, it has converged to a local optimum about 8–10 reward points below the rule-based ceiling. Simply running longer does not help, which is part of what motivates the Bayesian tuning study.

PPO matches or beats DQN on most profiles (207.6 on `activity_consistency`, 205.5 on `maintenance`) but triggers a pathological number of safety interventions on `gradual_management` (30.4/episode vs 0 for everything else). The reward still lands at 200 despite 30+ interventions because the safety penalty weight is modest and the positive components dominate — but the interventions indicate PPO is persistently trying to push calorie cuts that the safety layer clips. This is worth investigating and is documented in `docs/technical-challenges.md`.

The Bayesian tuning search (50 trials, Optuna TPE) found best DQN parameters: `lr=1.08e-4, gamma=0.937, batch_size=128, hidden_size=256, epsilon_decay=55`. The tuning history plot (`results/plots/tuning_history.png`) shows the expected pattern of early exploration followed by convergence in the profitable region of hyperparameter space.

**Episode inspection — `gradual_management` profile.** Running `scripts/inspect_episode.py --profile gradual_management` generates per-day trajectory plots across all policies for a single episode. Three consistent patterns emerge across three different seeds (7, 42, 99):

1. **PPO triggers 38–44 safety interventions** per episode and ends with the user at ~82.4 kg — *gaining* weight (trend ≈ +0.05 kg/7d) — despite the goal being slow weight loss. The action heatmap in the inspection plot shows PPO overwhelmingly picks calorie-increase actions that the safety layer clips. This is reward hacking: PPO maximizes adherence and recovery scores by keeping the simulated user comfortable while ignoring goal progress entirely.

2. **DQN and rule-based converge to virtually identical trajectories**: same final weight (~80.5 kg), same 7-day trend (≈ −0.16 kg/7d), zero safety interventions across all seeds. This confirms DQN has effectively learned the rule-based policy, which explains the small reward gap. The DQN is correct — it is just not discovering a strictly better strategy than the heuristic on this environment.

3. **Random** has the worst outcome: the user gains weight aggressively (+0.4 to +1.0 kg/7d by episode end) due to random calorie-increase actions not being consistently balanced by decreases.

These episode-level plots are saved to `results/plots/inspect_gradual_management_ep7.png`, `ep42.png`, and `ep99.png`.

The visual inspection confirms two findings from the aggregate metrics. First, DQN has converged to a near-identical policy as the rule-based heuristic — both choose "maintain" almost every day, achieve the same weight trend (−0.16 kg/7d), and generate zero safety clips. Second, PPO's failure on `gradual_management` is a case of reward hacking: it picks diverse actions that include calorie-increase and activity-increase proposals, gets those clipped 38+ times per episode, and still earns high adherence/recovery reward — so from the policy's perspective the behavior looks fine, but the user never achieves the weight-loss goal. The safety layer is working correctly (it blocks every unsafe proposal), but it is not penalizing the *pattern* strongly enough to force PPO to find a better policy.

**High-entropy PPO experiment.** To investigate whether more exploration could fix the `gradual_management` anomaly, PPO was retrained with `ent_coef=0.05` (vs the default 0.01) and evaluated under the same conditions. The result was unambiguously worse: interventions on `gradual_management` increased from 30.4 to **45.0/episode**, and intervention rates spiked on profiles that were previously clean (`activity_consistency`: 0.0 → 39.3; `maintenance`: 0.0 → 19.5; `performance_support`: 0.0 → 31.6). Rewards dropped 15–20 points across the board. Higher entropy did not help PPO discover a better policy — it made the policy more erratic everywhere. This rules out "not enough exploration" as the root cause. The more likely explanation is that the reward signal does not make goal-progress consequential enough relative to the adherence and recovery components to motivate a calorie-reduction strategy, regardless of exploration level. The fix would be to increase the `goal_progress` reward weight in `configs/env.yaml`, not to tune PPO's entropy. This is listed as future work.

**A note on the DQN 500-episode model.** Training DQN for 500 episodes (vs the default 250) produced a model with dramatically worse safety behavior: 39.5 interventions/episode on `general_wellness` vs 0.05 with the 250-episode model, at essentially the same reward. After ε hits its floor at episode 200, the agent had 300 more updates to discover that clipped calorie-cut proposals still accumulate reward from adherence and recovery — and it exploited this. The 250-episode model was restored for final evaluation. This is a concrete instance of reward hacking under a safety constraint: the penalty weight (`w_unsafe=1.5`) was not strong enough to deter the behavior once exploration pressure was gone. Full analysis in `docs/technical-challenges.md`.

## 10. Discussion

The most interesting design tension in this project was between the *agent's freedom to explore* and the *safety layer's authority to clip*. If the safety layer is too permissive, the agent can learn unsafe policies that look good on short-term reward. If the safety layer is too aggressive, the agent has no meaningful learning signal because every interesting action gets clipped to "maintain." The current configuration — calorie change capped at ±150 kcal/day, step increase capped at +1500/day, low-recovery override at 0.35 — gives the agent real but bounded freedom.

I spent more time than expected on the reward shaping. The initial penalty sign error (rewarding rather than penalizing unsafe actions, caught by `test_rewards.py`) was a good reminder that the reward function is where the most dangerous bugs hide in RL — the agent trains happily against a broken reward without any obvious error signal until you look at the actual behavior.

The five-profile library is small enough to keep the project tractable but large enough to expose some generalization. One thing I would do differently is use held-out test profiles for the evaluation — right now the agent is evaluated on the same distribution it trains on.

The DQN not consistently beating the rule-based after 250 training episodes is worth addressing honestly. The rule-based heuristic was designed with full knowledge of the reward structure (it explicitly checks adherence, recovery, sleep, and weight trend), which gives it an unfair advantage over the RL agents that have to infer this from the reward signal. That said, DQN does outperform the rule-based on `maintenance` and `performance_support`, and the training curve shows reward still climbing at episode 250, suggesting that 500–1000 episodes would likely close the gap on the remaining profiles. The Bayesian tuning results support this: the best found `epsilon_decay=55` is much shorter than the default 200, meaning exploration ends early and the agent exploits a partially learned policy for most of the 250 episodes.

## 11. Limitations

Already enumerated at length in `docs/decisions.md` and `docs/technical-challenges.md`. Brief summary:

- The simulator is a teaching simulator and is not clinically validated. The Mifflin-St Jeor BMR [8] and the 7700 kcal/kg constant are plausible approximations, not clinical parameters.
- The reward weights are hand-tuned, not learned. Inverse RL or preference-based RL would be the principled replacement.
- The action space is small and discrete (8 actions). A continuous parameterization would let the agent fine-tune the magnitude of each adjustment but at the cost of much harder exploration.
- The Bayesian tuning study covers only DQN. Running the same search on PPO's hyperparameters is listed as future work.

## 12. Conclusion

The project delivers a working, configurable, well-tested RL environment for safe lifestyle recommendation. If I had to pick the most valuable thing it demonstrates, it is not the agents — it is the combination of safety layer design and reward decomposition that makes the agents learnable in the first place. The safety layer keeps the agent out of pathological regions; the reward decomposition makes every component inspectable and ablatable.

The biggest contribution of the V3 update over V2 is the evaluation discipline: every policy runs against every profile, every plot is reproducible from the metrics CSV, and every architectural decision is justified and cited.

## References

Full citation details and a claim-to-reference traceability table are in [`docs/citations.md`](citations.md).

[1] Sutton, R. S. and Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.

[2] Mnih, V. et al. "Human-level control through deep reinforcement learning." *Nature* 518, 529–533 (2015).

[3] Mnih, V. et al. "Playing Atari with Deep Reinforcement Learning." arXiv:1312.5602 (2013).

[4] Schulman, J. et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017).

[4b] Schulman, J. et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv:1506.02438 (2015).

[5] Towers, M. et al. "Gymnasium: A Standard Interface for Reinforcement Learning Environments." arXiv:2407.17032 (2023).

[6] Raffin, A. et al. "Stable-Baselines3: Reliable Reinforcement Learning Implementations." *JMLR* 22(268), 2021.

[7] Akiba, T. et al. "Optuna: A Next-generation Hyperparameter Optimization Framework." KDD 2019.

[8] Mifflin, M. D. et al. "A new predictive equation for resting energy expenditure in healthy individuals." *AJCN* 51(2), 241–247 (1990).
