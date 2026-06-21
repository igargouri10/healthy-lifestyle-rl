"""Safety layer for HealthyLifestyleEnv.

The agent emits a **discrete action id** describing the type of adjustment it
wants to make.  The safety layer translates that into concrete numeric deltas
(calories, sleep hours, steps, protein g) and then *clips* those deltas against
configurable physiological bounds.  If a clip happens, the safety layer reports
the intervention back through the env's info dict so it can be penalized in
the reward and tracked in evaluation metrics.

This keeps the agent away from extreme calorie restriction, sudden activity
spikes, sleep deprivation, and over-aggressive protein targets, regardless of
what its Q-values or policy logits suggest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..simulation.transition_model import Deltas, DynamicState


# ---------------------------------------------------------------------------
# Action -> raw deltas mapping (the meaning of each discrete action id)
# ---------------------------------------------------------------------------
ACTION_MEANINGS: Dict[int, str] = {
    0: "maintain",
    1: "decrease calorie target slightly",
    2: "increase calorie target slightly",
    3: "increase activity (steps) slightly",
    4: "decrease activity / prioritize recovery",
    5: "increase sleep target",
    6: "raise protein within safe range",
    7: "rebalance toward moderate carbs",
}
N_ACTIONS = len(ACTION_MEANINGS)


# Per-action raw delta magnitudes BEFORE clipping.
RAW_DELTAS: Dict[int, Deltas] = {
    0: Deltas(),
    1: Deltas(calorie=-120.0),
    2: Deltas(calorie=+120.0),
    3: Deltas(steps=+1000),
    4: Deltas(steps=-1000),
    5: Deltas(sleep=+0.25),
    6: Deltas(protein=+10.0),
    7: Deltas(calorie=+60.0, protein=-5.0),  # tilt carbs up via remaining-kcal logic
}


@dataclass
class SafetyConfig:
    """Bounds used by the safety layer. All values configurable from YAML."""

    min_calorie_target: float = 1500.0
    max_calorie_target: float = 3500.0
    max_daily_calorie_change: float = 150.0
    min_sleep_target_h: float = 6.5
    max_sleep_target_h: float = 9.5
    max_step_increase_per_day: int = 1500
    min_recovery_threshold: float = 0.35
    min_protein_g_per_kg: float = 1.2
    max_protein_g_per_kg: float = 2.4

    @classmethod
    def from_dict(cls, d: Dict) -> "SafetyConfig":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


class SafetyLayer:
    """Converts discrete actions into clipped Deltas plus an info report."""

    def __init__(self, config: SafetyConfig):
        self.config = config

    def __call__(
        self, state: DynamicState, action: int, weight_kg: float,
    ) -> Tuple[Deltas, Dict]:
        cfg = self.config
        proposed = RAW_DELTAS.get(int(action), Deltas())

        # Start with the raw proposal.
        d = Deltas(
            calorie=proposed.calorie,
            sleep=proposed.sleep,
            steps=proposed.steps,
            protein=proposed.protein,
        )
        info = {
            "raw_action": int(action),
            "safe_action": int(action),
            "safety_intervention_applied": False,
            "unsafe_proposal": False,
            "clipped_components": [],
        }

        # --- Calorie bounds ----------------------------------------------------
        # 1) Per-day change cap
        if abs(d.calorie) > cfg.max_daily_calorie_change:
            d.calorie = float(
                cfg.max_daily_calorie_change if d.calorie > 0 else -cfg.max_daily_calorie_change
            )
            info["safety_intervention_applied"] = True
            info["clipped_components"].append("calorie_delta")

        # 2) Absolute calorie window
        new_cal_target = state.calorie_target + d.calorie
        if new_cal_target < cfg.min_calorie_target:
            d.calorie = cfg.min_calorie_target - state.calorie_target
            info["safety_intervention_applied"] = True
            info["unsafe_proposal"] = True
            info["clipped_components"].append("calorie_floor")
        elif new_cal_target > cfg.max_calorie_target:
            d.calorie = cfg.max_calorie_target - state.calorie_target
            info["safety_intervention_applied"] = True
            info["unsafe_proposal"] = True
            info["clipped_components"].append("calorie_ceiling")

        # --- Sleep bounds ------------------------------------------------------
        new_sleep_target = state.sleep_target_h + d.sleep
        if new_sleep_target < cfg.min_sleep_target_h:
            d.sleep = cfg.min_sleep_target_h - state.sleep_target_h
            info["safety_intervention_applied"] = True
            info["unsafe_proposal"] = True
            info["clipped_components"].append("sleep_floor")
        elif new_sleep_target > cfg.max_sleep_target_h:
            d.sleep = cfg.max_sleep_target_h - state.sleep_target_h
            info["safety_intervention_applied"] = True
            info["clipped_components"].append("sleep_ceiling")

        # --- Activity (steps) cap ---------------------------------------------
        if d.steps > cfg.max_step_increase_per_day:
            d.steps = int(cfg.max_step_increase_per_day)
            info["safety_intervention_applied"] = True
            info["unsafe_proposal"] = True
            info["clipped_components"].append("step_increase")
        # Don't allow a same-day cut to drop step target below zero.
        if state.step_target + d.steps < 0:
            d.steps = -int(state.step_target)
            info["safety_intervention_applied"] = True
            info["clipped_components"].append("step_floor")

        # --- Protein per kg of bodyweight -------------------------------------
        new_protein_target = state.protein_target_g + d.protein
        pmin = cfg.min_protein_g_per_kg * weight_kg
        pmax = cfg.max_protein_g_per_kg * weight_kg
        if new_protein_target < pmin:
            d.protein = pmin - state.protein_target_g
            info["safety_intervention_applied"] = True
            info["clipped_components"].append("protein_floor")
        elif new_protein_target > pmax:
            d.protein = pmax - state.protein_target_g
            info["safety_intervention_applied"] = True
            info["clipped_components"].append("protein_ceiling")

        # --- Low recovery override --------------------------------------------
        # If the user is under-recovered, suppress any further activity increase
        # and force a small recovery action.
        if state.recovery_score < cfg.min_recovery_threshold and d.steps > 0:
            d.steps = 0
            info["safety_intervention_applied"] = True
            info["unsafe_proposal"] = True
            info["clipped_components"].append("low_recovery_block")

        return d, info


def apply_safety_constraints(
    state: DynamicState, proposed_action: int, config: SafetyConfig, weight_kg: float,
) -> Tuple[Deltas, Dict]:
    """Functional wrapper for tests and standalone use."""
    return SafetyLayer(config)(state, proposed_action, weight_kg)
