"""Smoke / API tests for HealthyLifestyleEnv."""

from __future__ import annotations

import numpy as np
import pytest

from lifestyle_rl.envs.healthy_lifestyle_env import HealthyLifestyleEnv


def test_env_initializes():
    env = HealthyLifestyleEnv()
    assert env.action_space.n == 8
    assert env.observation_space.shape == (env.OBS_DIM,)


def test_reset_returns_valid_obs():
    env = HealthyLifestyleEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.OBS_DIM,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_step_returns_correct_tuple():
    env = HealthyLifestyleEnv()
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    for k in ("safe_action", "reward_components", "current_weight_kg",
              "adherence_score", "recovery_score", "sleep_hours", "steps"):
        assert k in info


def test_action_space_works():
    env = HealthyLifestyleEnv()
    env.reset(seed=1)
    for a in range(env.action_space.n):
        env.step(a)


def test_invalid_action_rejected():
    env = HealthyLifestyleEnv()
    env.reset(seed=2)
    with pytest.raises(ValueError):
        env.step(99)


def test_episode_terminates():
    env = HealthyLifestyleEnv()
    env.episode_days = 5  # shorten for the test
    env.reset(seed=3)
    done = False
    steps = 0
    while not done and steps < 50:
        _, _, term, trunc, _ = env.step(0)
        done = term or trunc
        steps += 1
    assert done
    assert steps == 5
