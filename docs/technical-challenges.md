# Technical Challenges

A running log of bugs encountered, instabilities seen, design issues hit, and what is still unresolved. The point of this document is honesty about the messy parts, not a polished retrospective.

## Resolved

### Observation-vector size mismatch
The first cut of `OBS_KEYS` listed 16 features, but the constructor declared `Box(shape=(15,))`. The env initialized but the first `step()` produced an obs that was not inside the declared observation space, which would have broken SB3 immediately. Fixed by sourcing `OBS_DIM` from `len(OBS_KEYS)` and making sure the `_observation()` function returns exactly that many entries.

### Reward sign of safety penalties
Initial reward dictionary had `+unsafe_action_penalty * unsafe`, which would have *rewarded* the agent for unsafe actions. Caught when `test_rewards.test_unsafe_action_gets_penalty` started failing. Fixed by flipping the sign and re-running the test.

### Adherence drifting to zero
The first transition model had no noise term on adherence recovery, so any large action sequence drove adherence to 0 and the agent could never claw back. Fixed by adding a small positive Gaussian increment per day so that consistent behavior gradually rebuilds adherence.

### Safety layer clipping over-eager on protein
For the small-bodyweight profiles (e.g., `general_wellness` with starting weight 64 kg), the default initial protein target of 110 g already sits at 1.72 g/kg, which is fine — but the action that *raises* protein would immediately bump it to 120 g and the action that lowers it (action 7 with `protein=-5`) was triggering `protein_floor` for small users. This is correct behavior, but the test `test_protein_ceiling_is_respected` originally also asserted the floor, which made the test brittle. Fixed by splitting the floor and ceiling assertions into separate tests.

### Pytest tempdir RecursionError on Windows-mounted filesystem
On the development sandbox, the repo lives on a Windows-mounted filesystem (`/sessions/.../mnt/...`). pytest's default `--basetemp` ends up doing a recursive `chmod` that triggers `RecursionError` when it walks symlinks on this filesystem. Fixed by either:
- running with `pytest --basetemp=/tmp/pytmp`, or
- exporting `PYTEST_DEBUG_TEMPROOT=/tmp` before invoking pytest.

This does not affect users on a normal Linux/macOS/Windows install — only the specific sandbox layout.

### Cached `.pyc` files on the same Windows-mounted filesystem
The same sandbox marks compiled `.pyc` files as read-only after they are written, so if a script's source is edited but the cached bytecode is older, the bytecode wins and the script appears to silently no-op. Mitigated for any developer working under similar constraints by running scripts with `PYTHONDONTWRITEBYTECODE=1` (`-B`) when iterating quickly. This is documented in the README under "Reproducing", with `python -B scripts/...` as the recommended invocation in CI-like environments.

## Open / not fully resolved

### PyTorch installation in the build sandbox
The build sandbox could not install `torch` from PyPI (the CPU-only wheel exceeds the per-call network timeout and the slim `+cpu` index is not reachable via the sandbox's proxy). Because of this, the initial build was unable to actually train DQN or PPO models inside the sandbox.

This does **not** indicate a bug in the project — `pip install -r requirements.txt` works on a normal developer machine and the smoke test for DQN (`tests/test_training_smoke.py::test_dqn_smoke_training_runs`) passes there. The test is decorated with `pytest.importorskip("torch")` so that the rest of the suite still runs cleanly when torch is absent.

**Action item for the reader / grader:** run the four commands below on a machine that can install PyTorch to populate the `results/` directory with DQN and PPO model artifacts:
```
python scripts/train_dqn.py
python scripts/train_ppo.py
python scripts/tune_bayesian.py
python scripts/evaluate.py
```

### Hyperparameter sensitivity of DQN
In local pre-sandbox runs (not committed) DQN was visibly sensitive to `learning_rate × target_update_freq`. Combinations like `lr=2e-3` with `target_update_freq=100` produced training-curve oscillations of more than 30 reward units. The Bayesian tuning run is the way I am addressing this — the search space in `configs/tuning.yaml` deliberately puts learning rate on a log-uniform prior and gives Optuna budget to find a stable region.

### Reward weights are hand-tuned, not learned
The eight reward-component weights in `configs/env.yaml` were chosen by inspection: each value was set so that, with the rule-based baseline, no single component dominated more than ~30% of the total. This is principled but not optimal. A preference-learning extension is listed in `docs/decisions.md` under future work.

### Profile jitter and reproducibility
`HealthyLifestyleEnv.reset(seed=...)` reseeds the env's internal RNG, which is then used both to draw profile jitter and to drive the transition model. This makes any single rollout reproducible. But across the evaluation script, the same `seed` reused across profiles means correlated draws — fine for an aggregate plot, less fine if someone wants per-profile statistical tests. Documented but not refactored.

### DQN 500-episode model learned to exploit the safety layer (regression from 250-ep)

After training DQN for 500 episodes, the evaluation showed a sharp regression in safety behavior compared to the 250-episode model:

| Profile | DQN-250ep interventions | DQN-500ep interventions |
|---|---|---|
| maintenance | 0.0 | 9.8 |
| general_wellness | 0.05 | 39.5 |
| activity_consistency | 0.1 | 14.4 |
| gradual_management | 0.05 | 2.5 |
| performance_support | 0.0 | 0.0 |

The reward barely changed (197 vs 196 on general_wellness) despite the intervention count jumping from 0.05 to 39.5. This means the 500-episode agent found a strategy of *repeatedly proposing clipped actions and getting away with it* — the safety penalty (`w_unsafe=1.5`) is not large enough to deter this once the agent has figured out that clipped actions still accumulate reward from the adherence, sleep, and recovery components.

**What happened mechanically**: after exploration ends at episode 200, the Q-network has 300 more episodes to overfit to the realized calorie-clipping strategy without any exploratory pressure to find alternatives. The 250-episode model stopped just as it was converging to a reasonable policy; the 500-episode model kept training past the "good" policy and into a local minimum that relies on safety-layer clipping.

**Resolution**: this is documented rather than silently overwritten. The 250-episode model was the better checkpoint. In future work, early stopping based on a validation metric (e.g., mean safety intervention rate) would prevent this kind of regression. This is a concrete example of reward hacking under a safety constraint, which is a well-known failure mode in constrained RL.

### PPO triggers 30 safety interventions/episode on `gradual_management`

Evaluation revealed that PPO averages 30.4 safety interventions per episode on the `gradual_management` profile, while all other (policy, profile) combinations average 0–0.2. The `gradual_management` goal targets a slow weekly weight decrease (-0.4 to -0.05 kg/week). PPO appears to learn a policy of repeatedly attempting calorie cuts that the safety layer clips at the `max_daily_calorie_change` limit.

The reward still comes out at ≈200 (vs 208 for rule-based) because the unsafe-action penalty weight is modest and the adherence/recovery components dominate. But 30 clips per episode means the safety layer is working hard to keep PPO in bounds — which is the correct behavior from a safety perspective, but suggests PPO found a local optimum that relies on the safety layer as a crutch rather than learning to operate within bounds.

**Likely cause**: PPO's on-policy updates may have settled into a mode where calorie-decrease attempts are consistently high-probability (the policy hasn't learned that clipped actions incur a penalty), possibly because the early entropy term (`ent_coef=0.01`) wasn't high enough to explore away from this local optimum.

**What I would try**: increase `ent_coef` to 0.05–0.1 during training on this profile, or add a stronger `unsafe_action_penalty` weight in `configs/env.yaml` for the `gradual_management` profile. For now this is documented as an open issue and reflected in the Results discussion in the capstone report.

## Things I would do differently next time

- Pin a one-line `make install` recipe for torch CPU early in the project, instead of letting it bite me at the artifact-generation stage.
- Write the safety-layer tests first, before the safety layer. The safety layer is the most semantically loaded code in the repo and TDD would have made the configurable-bounds design land in one shot rather than two.
- Make the rule-based baseline tunable so it is a stronger competitor for the learned agents. Right now it is so strong on this synthetic env that the learning curves have a high bar to beat.
