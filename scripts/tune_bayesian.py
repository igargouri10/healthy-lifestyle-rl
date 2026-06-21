"""Bayesian hyperparameter tuning for the DQN agent with Optuna.

Saves:
  results/metrics/bayesian_tuning_results.csv
  results/plots/tuning_history.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
import optuna

from lifestyle_rl.agents.dqn_agent import DQNAgent, DQNConfig
from lifestyle_rl.envs.healthy_lifestyle_env import HealthyLifestyleEnv
from lifestyle_rl.utils import set_global_seed
from lifestyle_rl.utils.plotting import plot_optuna_history


def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _suggest(trial: optuna.Trial, name: str, spec: Dict):
    t = spec["type"]
    if t == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if t == "uniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if t == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown distribution type: {t}")


def make_objective(search_space: Dict, episodes_per_trial: int, seed: int):
    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}
        dqn_cfg = DQNConfig(
            hidden_sizes=[int(params.get("hidden_size", 128))] * 2,
            gamma=float(params.get("gamma", 0.99)),
            learning_rate=float(params.get("learning_rate", 5e-4)),
            batch_size=int(params.get("batch_size", 64)),
            epsilon_decay_episodes=int(params.get("epsilon_decay_episodes", 200)),
            seed=seed,
        )
        set_global_seed(seed)
        env = HealthyLifestyleEnv()
        agent = DQNAgent(dqn_cfg)
        rng = np.random.default_rng(seed)

        rewards = []
        for ep in range(episodes_per_trial):
            obs, info = env.reset(seed=seed + ep)
            eps = agent.epsilon(ep)
            total = 0.0
            done = False
            while not done:
                a = agent.act(obs, epsilon=eps, rng=rng)
                next_obs, r, term, trunc, info = env.step(a)
                done = term or trunc
                agent.remember(obs, a, r, next_obs, done)
                agent.train_step()
                obs = next_obs
                total += r
            rewards.append(total)
        # Mean reward over the last 25% of episodes
        tail = max(5, episodes_per_trial // 4)
        return float(np.mean(rewards[-tail:]))

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tuning.yaml")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--episodes-per-trial", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tuning_cfg = cfg.get("tuning", {})
    n_trials = int(args.trials if args.trials is not None else tuning_cfg.get("n_trials", 15))
    epi = int(args.episodes_per_trial if args.episodes_per_trial is not None
              else tuning_cfg.get("episodes_per_trial", 60))
    seed = int(args.seed if args.seed is not None else tuning_cfg.get("seed", 0))
    storage = tuning_cfg.get("storage")
    study_name = tuning_cfg.get("study_name", "dqn_hparam_search")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction=tuning_cfg.get("direction", "maximize"),
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
    )
    objective = make_objective(cfg.get("search_space", {}), epi, seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    rows = []
    for t in study.trials:
        row = {"trial": t.number, "value": t.value}
        row.update(t.params)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = Path(tuning_cfg.get("results_csv", "results/metrics/bayesian_tuning_results.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    plot_optuna_history([t.value for t in study.trials if t.value is not None],
                        tuning_cfg.get("history_plot", "results/plots/tuning_history.png"))
    print(f"Best value: {study.best_value:.3f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved tuning results -> {csv_path}")


if __name__ == "__main__":
    main()
