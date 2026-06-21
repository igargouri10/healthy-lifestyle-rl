"""Smoke tests that exercise the DQN training loop and the evaluation
helpers for a handful of episodes only.

These tests intentionally use tiny budgets so they finish in a few seconds.
They do not assert convergence; they assert the code runs without crashing.
"""

from __future__ import annotations

import numpy as np
import pytest

from lifestyle_rl.agents.baselines import RandomPolicy, RuleBasedPolicy
from lifestyle_rl.envs.healthy_lifestyle_env import HealthyLifestyleEnv


def test_dqn_smoke_training_runs():
    torch = pytest.importorskip("torch")  # noqa: F841
    from lifestyle_rl.agents.dqn_agent import DQNAgent, DQNConfig
    env = HealthyLifestyleEnv()
    env.episode_days = 8  # very short
    cfg = DQNConfig(
        hidden_sizes=[32, 32],
        buffer_size=500,
        min_replay_size=20,
        epsilon_decay_episodes=5,
        batch_size=16,
        train_freq=1,
        target_update_freq=20,
    )
    agent = DQNAgent(cfg)
    rng = np.random.default_rng(0)
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        eps = agent.epsilon(ep)
        done = False
        while not done:
            a = agent.act(obs, epsilon=eps, rng=rng)
            next_obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            agent.remember(obs, a, r, next_obs, done)
            agent.train_step()
            obs = next_obs
    # If we got here without exceptions the smoke test has passed.
    assert agent._step_count > 0


def test_evaluation_one_episode_random():
    env = HealthyLifestyleEnv()
    env.episode_days = 5
    pol = RandomPolicy(seed=0)
    obs, info = env.reset(seed=0)
    total = 0.0
    done = False
    while not done:
        a = pol.act(obs, info)
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    assert isinstance(total, float)


def test_rule_based_one_episode():
    env = HealthyLifestyleEnv()
    env.episode_days = 5
    pol = RuleBasedPolicy()
    obs, info = env.reset(seed=0)
    done = False
    while not done:
        a = pol.act(obs, info)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
    assert info["safe_action"] in range(8)
