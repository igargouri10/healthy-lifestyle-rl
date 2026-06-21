"""Train a Stable-Baselines3 PPO agent on HealthyLifestyleEnv.

Outputs:
  results/trained_models/ppo_model.zip
  results/metrics/ppo_training_metrics.csv
  results/plots/ppo_training_curve.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from stable_baselines3.common.callbacks import BaseCallback

from lifestyle_rl.agents.ppo_agent import build_ppo
from lifestyle_rl.envs.healthy_lifestyle_env import HealthyLifestyleEnv
from lifestyle_rl.utils import set_global_seed
from lifestyle_rl.utils.plotting import plot_training_curve


class EpisodeRewardLogger(BaseCallback):
    """Captures completed episode rewards from SB3's `infos`."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: List[float] = []
        self._running = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        if rewards is None or dones is None:
            return True
        for r, d in zip(np.atleast_1d(rewards), np.atleast_1d(dones)):
            self._running += float(r)
            if bool(d):
                self.episode_rewards.append(self._running)
                self._running = 0.0
        return True


def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config).get("ppo", {})
    timesteps = int(args.timesteps if args.timesteps is not None else cfg.get("total_timesteps", 30000))
    cfg["seed"] = args.seed
    set_global_seed(args.seed)

    env = HealthyLifestyleEnv()
    model = build_ppo(env, cfg)
    cb = EpisodeRewardLogger()
    model.learn(total_timesteps=timesteps, callback=cb)

    save_path = cfg.get("save_path", "results/trained_models/ppo_model.zip")
    metrics_path = cfg.get("metrics_path", "results/metrics/ppo_training_metrics.csv")
    plot_path = cfg.get("plot_path", "results/plots/ppo_training_curve.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)

    rewards = cb.episode_rewards
    pd.DataFrame({"episode": np.arange(len(rewards)), "reward": rewards}).to_csv(
        metrics_path, index=False,
    )
    if rewards:
        plot_training_curve(rewards, plot_path, title="PPO training curve")
    print(f"Saved model -> {save_path}")
    print(f"Saved metrics -> {metrics_path}  ({len(rewards)} episodes)")
    print(f"Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
