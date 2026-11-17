# Healthy Lifestyle RL

**Safe Deep Reinforcement Learning for Personalized Nutrition, Sleep, and Activity Recommendation**

Healthy Lifestyle RL is a reinforcement learning capstone project that models daily lifestyle planning as a sequential decision-making problem. The project introduces a custom Gymnasium-compatible environment where agents learn safe recommendation policies for nutrition, sleep, and activity across simulated user profiles. The repository compares random, rule-based, DQN, and PPO agents, includes Bayesian hyperparameter tuning, and documents why certain advanced RL methods such as multi-agent coordination and robotics simulators were not selected for this single-agent environment.

This is a course capstone for EECS 590 (Reinforcement Learning).

## Safety disclaimer

**This project is for simulation and academic RL research only. It is not a medical tool and the trained agents do not provide medical advice.** The simulated transition model is intentionally simple and is not validated against any clinical data. Any reference to calories, sleep, activity, or body weight is a *simulation parameter inside an environment*. The configurable safety layer keeps the agent inside a sensible behavioral envelope (no extreme restriction, no sudden activity jumps, no sleep deprivation, no overshoot on protein intake), but it is not a substitute for professional medical guidance.

## Why this is an RL problem

Daily lifestyle planning is a sequential decision problem with delayed feedback, partial observability, and a need to balance multiple long-horizon objectives (adherence, recovery, goal progress, safety). A myopic rule-based planner can reach the lowest-hanging fruit but cannot learn the compounding effects of recovery debt, adherence drift, or progressive overload. The MDP formulation gives the agent the credit assignment it needs to schedule small daily adjustments whose value only shows up days later.

## MDP formulation

- **States** (18-dim, normalized): age, sex code, height, initial weight, goal type, plus the dynamic features current weight, 7-day weight trend, calorie target, actual calorie intake, protein/carbs/fat, sleep hours, daily steps, recovery score, adherence score, energy score, and a normalized day index.
- **Actions** (8 discrete): `maintain`, `decrease calorie target`, `increase calorie target`, `increase activity`, `prioritize recovery`, `increase sleep target`, `raise protein`, `rebalance toward carbs`. The agent never sets absolute values — only adjustments, which the safety layer then clips.
- **Transition** ``T(s, a) -> s'``: implemented in `src/lifestyle_rl/simulation/transition_model.py`. A combination of deterministic adherence-weighted execution of targets plus Gaussian noise on intake, sleep, and steps; slow weight drift from cumulative calorie balance; recovery and energy scores driven by sleep and load.
- **Reward**: decomposed into eight weighted components (goal progress, adherence, sleep, activity, recovery, unsafe-action penalty, rapid-change penalty, instability penalty). All weights live in `configs/env.yaml`.
- **Horizon**: 60-day episodes by default (`episode_days` configurable: 30 quick / 60 medium / 90 final eval).
- **Discount**: `gamma = 0.99`.

## Repository structure

```
healthy-lifestyle-rl/
├── README.md                LICENSE                 .gitignore
├── pyproject.toml           requirements.txt
├── configs/                 env.yaml, dqn.yaml, ppo.yaml, tuning.yaml
├── src/lifestyle_rl/
│   ├── envs/                healthy_lifestyle_env.py
│   ├── agents/              baselines.py, dqn_agent.py, ppo_agent.py, replay_buffer.py
│   ├── rewards/             reward_functions.py
│   ├── safety/              safety_layer.py
│   ├── simulation/          user_profiles.py, transition_model.py
│   └── utils/               seeding.py, metrics.py, plotting.py
├── scripts/                 run_baselines.py, train_dqn.py, train_ppo.py,
│                            tune_bayesian.py, evaluate.py
├── results/                 plots/, metrics/, trained_models/
├── docs/                    decisions.md, technical-challenges.md, citations.md, capstone-report.md
└── tests/                   test_env.py, test_rewards.py, test_safety.py, test_training_smoke.py
```

## Installation

```bash
git clone <repo-url>
cd healthy-lifestyle-rl

python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows PowerShell

pip install -r requirements.txt
```

Python 3.10 or 3.11 is recommended.

## How to reproduce

```bash
# 1. sanity check
pytest

# 2. baselines (random + rule-based across all profiles)
python scripts/run_baselines.py

# 3. train the custom-PyTorch DQN agent
python scripts/train_dqn.py

# 4. train the Stable-Baselines3 PPO agent
python scripts/train_ppo.py

# 5. Bayesian hyperparameter search over DQN
python scripts/tune_bayesian.py

# 6. final evaluation: random vs rule-based vs DQN vs PPO across all profiles
python scripts/evaluate.py
```

Each script accepts CLI overrides — see `python scripts/<name>.py --help` for details.

## Results summary

Baselines run cleanly across all five profiles. The rule-based heuristic consistently outperforms the random policy (≈ 207 vs ≈ 178 mean episode reward on a 60-day horizon, 20-episode eval) and triggers essentially zero safety interventions, confirming that the safety layer correctly never has to clip a well-behaved policy. The random policy averages ≈ 3 safety-layer interventions per 60-day episode, demonstrating the safety layer is also being exercised. Full per-profile numbers are in `results/metrics/baselines_summary.csv` and `results/metrics/evaluation_summary.csv`.

DQN and PPO training scripts are fully implemented and tested for correctness via smoke tests; trained model artifacts are generated locally when the user has PyTorch installed (see `docs/technical-challenges.md` for the sandbox limitation that prevented PyTorch installation during the initial build).

## Documentation

- [`docs/decisions.md`](docs/decisions.md) — what was implemented, what was rejected, and why.
- [`docs/technical-challenges.md`](docs/technical-challenges.md) — bugs, instability, unresolved issues.
- [`docs/citations.md`](docs/citations.md) — papers, libraries, and people that informed the project.
- [`docs/capstone-report.md`](docs/capstone-report.md) — the full V3 report with abstract, methods, and results.

## Citations & acknowledgments

This repository was built as part of EECS 590 (Reinforcement Learning). It draws on:

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
- Mnih et al., "Human-level control through deep reinforcement learning," *Nature* 518 (2015) — DQN.
- Schulman et al., "Proximal Policy Optimization Algorithms," 2017 — PPO.
- Brockman et al., *OpenAI Gym*, arXiv:1606.01540 (2016); Towers et al., *Gymnasium*, arXiv:2407.17032 (2023).
- Raffin et al., *Stable-Baselines3*, JMLR 22(268), 2021.
- Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*.

Full BibTeX-style citations are in [`docs/citations.md`](docs/citations.md).

## Tools / chatbot disclosure

Class collaboration: course lectures (Slides 1-6) and the project prompt provided by the instructor.

AI assistance: Anthropic's Claude was used as a pair-programming and drafting assistant during construction of this repository, including for boilerplate scaffolding (config YAML, package `__init__.py` layouts), draft text in `docs/`, and refactoring suggestions. All architectural and algorithmic decisions (single-agent framing, choice to use SB3 for PPO and custom PyTorch for DQN, safety-layer design, profile library, rejection of multi-agent / robotics methods) are the author's. Any code suggested by the assistant was reviewed, edited, and tested by the author.

## License

MIT — see [LICENSE](LICENSE).

## Local recording workflow (October 2026)

A recording workflow has been prepared in the parent workspace's `DEMO_GUIDE.md`.
These are bounded demonstrations, not a claim that every experimental workflow has been reproduced.
Keep AI-assisted implementation attribution and the documented limitations in public descriptions.
