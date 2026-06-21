# Project Decisions

This document explains what I chose to implement in the Version-3 capstone, what I deliberately left out, and why. The professor explicitly asked for this kind of transparency, so I want it to feel like a justification, not a checklist.

## Implemented

- **Custom Gymnasium environment** (`HealthyLifestyleEnv`) with 18-dim normalized observations and 8 discrete adjustment actions. The agent never sets absolute calories, sleep, or steps — it can only push the current targets in a direction, and the safety layer clips those pushes.
- **Single-agent MDP formulation** with a 30/60/90-day episode length and discount 0.99.
- **Simulated user profiles** — five personas (maintenance, general wellness, activity consistency, gradual management, performance support) drawn from a small library with optional Gaussian jitter on age/height/weight/baseline activity, generated reproducibly via the env's seeded RNG.
- **Stochastic transition model** with adherence-weighted execution of targets, Gaussian noise on intake/sleep/steps, slow weight drift from cumulative calorie balance, and explicit recovery/energy/adherence dynamics.
- **Configurable safety layer** with calorie floor/ceiling, per-day calorie change cap, sleep window, max step-increase, protein-per-kg bounds, and a low-recovery override that blocks further activity increases.
- **Reward decomposition** into eight named components (goal progress, adherence, sleep, activity, recovery, unsafe-action penalty, rapid-change penalty, instability penalty), all weights configurable from YAML and surfaced in `info["reward_components"]` for ablation studies.
- **Two non-learning baselines** — uniform random and a fitness-planner heuristic — so the RL agents have something meaningful to beat.
- **Custom-PyTorch DQN** with experience replay, target network, epsilon-greedy decay, gradient clipping, save/load. I chose to write this from scratch (rather than use SB3's DQN) because the course material focused on Q-learning derivations and I wanted the implementation to mirror that derivation.
- **Stable-Baselines3 PPO** wrapped around our custom env. I chose SB3 for PPO specifically because PPO's correctness (clipped surrogate, GAE, value-function loss balancing) is easy to get subtly wrong and the course did not require me to re-implement it from scratch. Writing one solid algorithm by hand and using a battle-tested library for the other gives a more honest project than two half-baked custom implementations.
- **Bayesian hyperparameter tuning with Optuna** over learning rate, gamma, batch size, hidden size, and epsilon-decay length on the DQN agent.
- **Evaluation harness** that scores every policy on every profile, computes mean reward, safety-intervention rate, adherence, and action distribution, and saves four comparison plots.
- **Pytest suite** covering env API, reward shape and signs, safety-layer clipping behavior, and a torch-aware DQN smoke test.

## Not implemented (and why)

### Multi-agent RL — rejected
The environment models a single user's lifestyle optimization. There is no second agent to cooperate or compete with. Adding a second agent would require inventing a second principal (a coach? a household member?) which would change the research question.

### Swarm / coordination — rejected
There is no shared physical space, no group of robots, and no consensus problem. Importing a swarm framework would be dead weight.

### Isaac Sim, MuJoCo, PyBullet, or any robotics simulator — rejected
The environment is behavioral and physiological (calorie balance, recovery, adherence), not a physics-based control task. The agent does not actuate joints; it picks one of eight discrete recommendations per day. Adding a physics engine would slow training by orders of magnitude with no analytical benefit.

### PettingZoo, QMIX, MADDPG — rejected
All three are tools for multi-agent settings. They don't have anything to offer a single-agent MDP.

### Full offline RL (CQL, IQL, BCQ) — rejected for this version
Offline RL is the natural choice when you have a large fixed log of real user trajectories and the deployment cost of bad actions is high. I have *neither* condition here: my "data" is a synthetic simulator I built myself, and the trained agents are never deployed against real users. Offline RL is therefore not the right tool right now. It would become the right tool if anonymized user-tracker logs (e.g., from a consenting health-tracker cohort) became available, and that is listed under future work.

### Full Sim2Real transfer — rejected for this version
The project does not produce recommendations for real humans. Sim2Real transfer for a behavioral simulator like this one would require months of clinical validation before it could be ethical, and that is outside the scope of an undergraduate capstone.

### Meta-RL — out of scope (future work)
Meta-RL (e.g., MAML, RL^2) could in principle help a single agent adapt quickly to a previously unseen user profile. It is interesting but is not required by the current setup, where the profile is observable in the state vector and the agent already gets episode-level training across all five personas.

### Hierarchical RL — out of scope (future work)
A two-level hierarchy (weekly planner over a daily executor) is an attractive way to separate strategic from tactical decisions. I left it out because the current 8-action discrete space is already interpretable and the prompt explicitly asked me to avoid adding scope for its own sake.

## Why DQN

DQN is the discrete-action algorithm the course derived in detail. With only 8 actions, a relatively low-dimensional observation (18 features), and no constraints requiring stochastic policies, DQN is a natural fit. It also exposes the Bellman target and target-network cadence directly, which makes the Bayesian tuning study more interesting (you can see how `gamma` and `target_update_freq` interact).

## Why PPO

PPO is the on-policy counterpart most students see early in deep RL coursework. It complements DQN by giving us a stochastic actor and an explicit value baseline, and it tends to be more stable than DQN under modest reward shaping. Crucially, PPO supports continuous action spaces — so if a future iteration moves from discrete adjustments to continuous "set the calorie target to X" actions, the PPO pipeline transfers; the DQN pipeline does not. PPO is therefore the natural growth path.

## Why Bayesian (Optuna) tuning

The prompt specifically calls out that "an environment centered on a main protagonist does not necessarily benefit from swarm/coordination implementations, but very much benefits from Bayesian hyperparameter tuning." That matches the situation here precisely. A grid search over 5 hyperparameters with 5 values each is 3125 trial runs; random search wastes most of that budget; Bayesian (TPE) tuning concentrates trials on the regions where the surrogate model says the next trial is most informative. With episodes that take minutes rather than seconds, that sample efficiency matters.

## Safety choices

The safety layer sits *between* the agent and the transition model. The agent emits a discrete action id; the safety layer translates it into per-day deltas (calories, sleep, steps, protein) and clips those deltas against configurable bounds. Three design choices are worth calling out:

1. The agent cannot disable the safety layer. There is no action it can take that bypasses clipping, even if the Q-value for that action is high.
2. Every clip is reported through `info["clipped_components"]` and contributes an `unsafe_action_penalty` to the reward, so the agent has a learning signal to stay inside the safe region voluntarily.
3. The low-recovery override forces a recovery action when the recovery score drops below `min_recovery_threshold`, even if the agent wanted to push more activity. This is the safety property I would most defend in a code review.

## Limitations

- The transition model is a *teaching simulator*, not a clinical one. Maintenance calorie estimates use Mifflin-St Jeor with a fixed 1.4 activity factor; weight evolution uses the standard 7700 kcal/kg constant; sleep and recovery dynamics are heuristic. The numbers should be read as units inside the env, not as nutritional advice.
- The action space is small (8 actions). A continuous parameterization would let the agent fine-tune the magnitude of each adjustment but at the cost of much harder exploration.
- Reward weights were picked by inspection rather than learned from real preference data. Inverse RL or preference-based RL would be the principled replacement.

## Future work

- Run Bayesian tuning on PPO in addition to DQN.
- Add a preference-learning loop where the reward weights are themselves fit to held-out trajectories.
- Replace the synthetic simulator with a hybrid simulator + small real-data calibration loop.
- Introduce a hierarchical agent that picks a weekly objective and a daily executor.
- Add a continuous-action variant for the PPO pipeline.
