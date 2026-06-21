"""Train the custom DQN agent on HealthyLifestyleEnv.

Outputs:
  results/trained_models/dqn_model.pt
  results/metrics/dqn_training_metrics.csv
  results/plots/dqn_training_curve.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from lifestyle_rl.agents.dqn_agent import DQNAgent, DQNConfig
from lifestyle_rl.envs.healthy_lifestyle_env import HealthyLifestyleEnv
from lifestyle_rl.utils import set_global_seed
from lifestyle_rl.utils.plotting import plot_training_curve


def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def train(
    dqn_cfg: DQNConfig,
    total_episodes: int,
    seed: int,
    save_path: str,
    metrics_path: str,
    plot_path: str,
    log_every: int = 10,
) -> Dict:
    set_global_seed(seed)
    env = HealthyLifestyleEnv()
    agent = DQNAgent(dqn_cfg)
    rng = np.random.default_rng(seed)

    rewards, losses = [], []
    for ep in range(total_episodes):
        obs, info = env.reset(seed=seed + ep)
        eps = agent.epsilon(ep)
        total_r = 0.0
        done = False
        while not done:
            a = agent.act(obs, epsilon=eps, rng=rng)
            next_obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            agent.remember(obs, a, r, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            total_r += r
        rewards.append(total_r)
        if (ep + 1) % log_every == 0:
            recent = np.mean(rewards[-log_every:])
            print(f"ep={ep+1:>4d}/{total_episodes}  eps={eps:.2f}  "
                  f"recent_reward={recent:7.2f}  recent_loss={agent.recent_loss:.4f}")

    # Save artifacts
    agent.save(save_path)
    df = pd.DataFrame({"episode": np.arange(len(rewards)), "reward": rewards})
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_path, index=False)
    plot_training_curve(rewards, plot_path, title="DQN training curve")
    print(f"Saved model -> {save_path}")
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved plot -> {plot_path}")
    return {"final_avg_reward": float(np.mean(rewards[-max(10, len(rewards) // 10):]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dqn.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config).get("dqn", {})
    episodes = int(args.episodes if args.episodes is not None else cfg.get("total_episodes", 250))
    dqn_cfg = DQNConfig(
        hidden_sizes=list(cfg.get("hidden_sizes", [128, 128])),
        gamma=float(cfg.get("gamma", 0.99)),
        learning_rate=float(cfg.get("learning_rate", 5e-4)),
        batch_size=int(cfg.get("batch_size", 64)),
        buffer_size=int(cfg.get("buffer_size", 50000)),
        min_replay_size=int(cfg.get("min_replay_size", 1000)),
        epsilon_start=float(cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(cfg.get("epsilon_end", 0.05)),
        epsilon_decay_episodes=int(cfg.get("epsilon_decay_episodes", 200)),
        target_update_freq=int(cfg.get("target_update_freq", 500)),
        train_freq=int(cfg.get("train_freq", 4)),
        grad_clip=float(cfg.get("grad_clip", 10.0)),
        device=str(cfg.get("device", "cpu")),
        seed=args.seed,
    )
    train(
        dqn_cfg=dqn_cfg,
        total_episodes=episodes,
        seed=args.seed,
        save_path=cfg.get("save_path", "results/trained_models/dqn_model.pt"),
        metrics_path=cfg.get("metrics_path", "results/metrics/dqn_training_metrics.csv"),
        plot_path=cfg.get("plot_path", "results/plots/dqn_training_curve.png"),
        log_every=int(cfg.get("log_every", 10)),
    )


if __name__ == "__main__":
    main()
