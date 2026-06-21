"""Reward function for the HealthyLifestyleEnv.

The reward is a *decomposition* (each component is logged separately in
`info["reward_components"]`) so that ablations and tests are straightforward.

reward = + goal_progress + adherence + sleep + activity + recovery
        - unsafe_action_penalty - rapid_change_penalty - instability_penalty

Weights live in `configs/env.yaml` under the `rewards:` block and are loaded
into `RewardConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..simulation.transition_model import DynamicState, Deltas
from ..simulation.user_profiles import UserProfile


@dataclass
class RewardConfig:
    goal_progress: float = 1.0
    adherence: float = 0.8
    sleep: float = 0.6
    activity: float = 0.5
    recovery: float = 0.7
    unsafe_action_penalty: float = 1.5
    rapid_change_penalty: float = 0.6
    instability_penalty: float = 0.4

    @classmethod
    def from_dict(cls, d: Dict) -> "RewardConfig":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


# Per-goal target ranges. These describe "good behavior" for each goal type,
# NOT clinical targets. The agent is rewarded for *staying near* these ranges
# rather than for pushing extreme outcomes.
GOAL_TARGETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "maintenance":          {"weight_delta_per_week": (-0.1, 0.1)},
    "general_wellness":     {"weight_delta_per_week": (-0.2, 0.2)},
    "activity_consistency": {"weight_delta_per_week": (-0.3, 0.3)},
    "gradual_management":   {"weight_delta_per_week": (-0.4, -0.05)},
    "performance_support":  {"weight_delta_per_week": (-0.1, 0.2)},
}


def _bell(value: float, target_lo: float, target_hi: float, sigma: float) -> float:
    """1.0 inside [target_lo, target_hi]; falls off smoothly outside."""
    if target_lo <= value <= target_hi:
        return 1.0
    d = min(abs(value - target_lo), abs(value - target_hi))
    return float(np.exp(-(d ** 2) / (2 * sigma ** 2)))


def compute_reward(
    state: DynamicState,
    profile: UserProfile,
    deltas: Deltas,
    safety_info: Dict,
    cfg: RewardConfig,
) -> Tuple[float, Dict[str, float]]:
    """Return scalar reward plus per-component breakdown."""

    # --- Goal progress -----------------------------------------------------
    # The trend is per-7-days; positive=gain, negative=loss.
    trend = state.weight_trend_7d
    targets = GOAL_TARGETS.get(profile.goal_type, GOAL_TARGETS["general_wellness"])
    lo, hi = targets["weight_delta_per_week"]
    goal_progress = _bell(trend, lo, hi, sigma=0.4)

    # --- Behavioral consistency rewards ------------------------------------
    adherence = float(state.adherence_score)
    sleep_score = _bell(state.sleep_hours, 7.0, 8.5, sigma=0.6)
    activity_score = _bell(
        state.steps,
        max(profile.activity_baseline_steps - 1500, 3000),
        profile.activity_baseline_steps + 2500,
        sigma=2000,
    )
    recovery_score = float(state.recovery_score)

    # --- Penalties ---------------------------------------------------------
    unsafe = float(safety_info.get("unsafe_proposal", False))
    # Rapid change penalty proportional to total absolute target movement today.
    change_magnitude = (
        abs(deltas.calorie) / 150.0
        + abs(deltas.sleep) / 0.5
        + abs(deltas.steps) / 1500.0
        + abs(deltas.protein) / 20.0
    )
    rapid_change = max(0.0, change_magnitude - 1.0)  # only penalize >1 unit movement

    # Instability: did the per-day weight swing wildly?
    instability = 0.0
    if len(state.weight_history) >= 2:
        daily = abs(state.weight_history[-1] - state.weight_history[-2])
        instability = max(0.0, daily - 0.15)

    components = {
        "goal_progress":      cfg.goal_progress * goal_progress,
        "adherence":          cfg.adherence * adherence,
        "sleep":              cfg.sleep * sleep_score,
        "activity":           cfg.activity * activity_score,
        "recovery":           cfg.recovery * recovery_score,
        "unsafe_action_pen": -cfg.unsafe_action_penalty * unsafe,
        "rapid_change_pen":  -cfg.rapid_change_penalty * rapid_change,
        "instability_pen":   -cfg.instability_penalty * instability,
    }
    reward = float(sum(components.values()))
    return reward, components
