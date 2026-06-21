"""Tests for the reward decomposition."""

from __future__ import annotations

import numpy as np

from lifestyle_rl.rewards.reward_functions import RewardConfig, compute_reward
from lifestyle_rl.simulation.transition_model import Deltas, TransitionModel
from lifestyle_rl.simulation.user_profiles import PROFILE_LIBRARY


def _setup():
    profile = PROFILE_LIBRARY["general_wellness"]
    tm = TransitionModel()
    state = tm.init_state(profile, np.random.default_rng(0))
    return profile, state


def test_reward_is_numeric_and_decomposed():
    profile, state = _setup()
    cfg = RewardConfig()
    r, comps = compute_reward(state, profile, Deltas(), {"unsafe_proposal": False}, cfg)
    assert isinstance(r, float)
    assert isinstance(comps, dict)
    assert set(comps.keys()) >= {
        "goal_progress", "adherence", "sleep", "activity", "recovery",
        "unsafe_action_pen", "rapid_change_pen", "instability_pen",
    }
    assert abs(sum(comps.values()) - r) < 1e-5


def test_unsafe_action_gets_penalty():
    profile, state = _setup()
    cfg = RewardConfig()
    r_ok, _ = compute_reward(state, profile, Deltas(), {"unsafe_proposal": False}, cfg)
    r_bad, _ = compute_reward(state, profile, Deltas(), {"unsafe_proposal": True}, cfg)
    assert r_bad < r_ok


def test_high_adherence_gives_higher_reward_than_low():
    profile, state = _setup()
    cfg = RewardConfig()
    state.adherence_score = 0.95
    r_high, _ = compute_reward(state, profile, Deltas(), {}, cfg)
    state.adherence_score = 0.2
    r_low, _ = compute_reward(state, profile, Deltas(), {}, cfg)
    assert r_high > r_low


def test_low_recovery_is_penalized():
    profile, state = _setup()
    cfg = RewardConfig()
    state.recovery_score = 0.9
    r_high, _ = compute_reward(state, profile, Deltas(), {}, cfg)
    state.recovery_score = 0.1
    r_low, _ = compute_reward(state, profile, Deltas(), {}, cfg)
    assert r_high > r_low


def test_large_change_triggers_rapid_change_penalty():
    profile, state = _setup()
    cfg = RewardConfig()
    _, small = compute_reward(state, profile, Deltas(calorie=20), {}, cfg)
    _, big = compute_reward(state, profile, Deltas(calorie=600, steps=5000, sleep=2.0), {}, cfg)
    assert big["rapid_change_pen"] < small["rapid_change_pen"]
