"""Thin wrapper around Stable-Baselines3 PPO.

The PPO training loop is implemented in SB3; this module just packages a
constructor that reads our `configs/ppo.yaml` and produces a configured PPO
model bound to our custom environment.

The decision to use SB3 rather than a hand-rolled PPO is documented in
``docs/decisions.md``.
"""

from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
from stable_baselines3 import PPO


def build_ppo(env: gym.Env, ppo_cfg: Dict[str, Any]) -> PPO:
    """Construct a PPO model from a config dict (the `ppo:` block of ppo.yaml)."""
    return PPO(
        policy=ppo_cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 512)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        seed=int(ppo_cfg.get("seed", 0)),
        verbose=int(ppo_cfg.get("verbose", 0)),
    )
