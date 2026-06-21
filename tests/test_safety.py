"""Tests for the configurable safety layer."""

from __future__ import annotations

import numpy as np

from lifestyle_rl.safety.safety_layer import SafetyConfig, apply_safety_constraints
from lifestyle_rl.simulation.transition_model import TransitionModel
from lifestyle_rl.simulation.user_profiles import PROFILE_LIBRARY


def _state():
    profile = PROFILE_LIBRARY["general_wellness"]
    return profile, TransitionModel().init_state(profile, np.random.default_rng(0))


def test_safe_action_passes_unchanged():
    _, state = _state()
    cfg = SafetyConfig()
    deltas, info = apply_safety_constraints(state, 0, cfg, state.current_weight_kg)
    assert deltas.calorie == 0
    assert deltas.steps == 0
    assert info["safety_intervention_applied"] is False
    assert info["unsafe_proposal"] is False


def test_calorie_change_is_clipped():
    _, state = _state()
    cfg = SafetyConfig(max_daily_calorie_change=50)  # very tight
    deltas, info = apply_safety_constraints(state, 2, cfg, state.current_weight_kg)
    assert abs(deltas.calorie) <= 50
    assert info["safety_intervention_applied"] is True


def test_calorie_floor_blocks_unsafe_drop():
    _, state = _state()
    state.calorie_target = 1550  # very near min
    cfg = SafetyConfig(min_calorie_target=1500, max_daily_calorie_change=300)
    deltas, info = apply_safety_constraints(state, 1, cfg, state.current_weight_kg)
    assert state.calorie_target + deltas.calorie >= cfg.min_calorie_target
    assert info["unsafe_proposal"] is True


def test_low_recovery_blocks_activity_increase():
    _, state = _state()
    state.recovery_score = 0.1
    cfg = SafetyConfig(min_recovery_threshold=0.35)
    deltas, info = apply_safety_constraints(state, 3, cfg, state.current_weight_kg)
    assert deltas.steps <= 0
    assert info["safety_intervention_applied"] is True
    assert "low_recovery_block" in info["clipped_components"]


def test_protein_ceiling_is_respected():
    _, state = _state()
    state.protein_target_g = state.current_weight_kg * 2.39  # already near ceiling
    cfg = SafetyConfig(max_protein_g_per_kg=2.4)
    deltas, info = apply_safety_constraints(state, 6, cfg, state.current_weight_kg)
    assert state.protein_target_g + deltas.protein <= cfg.max_protein_g_per_kg * state.current_weight_kg + 1e-6
