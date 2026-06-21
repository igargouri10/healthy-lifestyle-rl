"""Day-by-day transition dynamics for the simulated user.

The model is deliberately simple, transparent, and stochastic.  It is *not*
a medical model; it is a teaching simulator for reinforcement learning.

State updated each day:

  * current_weight_kg
  * weight_trend_7d            (running 7-day delta)
  * actual_calorie_intake      (noisy realization of the calorie target)
  * protein_g, carbs_g, fat_g  (macro split, lightly noisy)
  * sleep_hours                (noisy around sleep_target)
  * steps                      (noisy around step_target)
  * training_load              (rolling avg of recent activity)
  * recovery_score             (0-1, depends on sleep and load)
  * energy_score               (0-1, depends on intake and recovery)
  * adherence_score            (0-1, drops when the agent jumps targets)
  * day_index
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, List, Optional
from collections import deque

import numpy as np

from .user_profiles import UserProfile


@dataclass
class DynamicState:
    """Mutable state that the transition model evolves day-to-day."""

    # Recommendation targets that the agent indirectly controls
    calorie_target: float = 2200.0
    sleep_target_h: float = 7.5
    step_target: int = 7000
    protein_target_g: float = 110.0

    # Realized / observed values
    current_weight_kg: float = 75.0
    weight_history: Deque[float] = field(default_factory=lambda: deque(maxlen=7))
    actual_calorie_intake: float = 2200.0
    protein_g: float = 110.0
    carbs_g: float = 250.0
    fat_g: float = 70.0
    sleep_hours: float = 7.5
    steps: int = 7000
    training_load: float = 0.5  # 0-1
    recovery_score: float = 0.7
    energy_score: float = 0.7
    adherence_score: float = 0.8
    day_index: int = 0

    @property
    def weight_trend_7d(self) -> float:
        if len(self.weight_history) < 2:
            return 0.0
        return float(self.weight_history[-1] - self.weight_history[0])


class TransitionModel:
    """Stateless evolver: takes (state, profile, deltas, rng) and steps one day.

    `deltas` describe the *requested adjustment* for that day, already filtered
    through the safety layer.  The agent does not set absolute values directly.
    """

    KCAL_PER_KG = 7700.0  # very rough energy balance constant for simulation

    def __init__(self, max_adherence_drop: float = 0.08):
        self.max_adherence_drop = max_adherence_drop

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init_state(self, profile: UserProfile, rng: np.random.Generator) -> DynamicState:
        maintenance = profile.maintenance_calories()
        # Start each user near their maintenance baseline.
        state = DynamicState(
            calorie_target=float(np.round(maintenance / 50) * 50),
            sleep_target_h=profile.sleep_baseline_h,
            step_target=profile.activity_baseline_steps,
            protein_target_g=float(np.round(profile.initial_weight_kg * 1.4)),
            current_weight_kg=profile.initial_weight_kg,
            actual_calorie_intake=maintenance,
            protein_g=float(np.round(profile.initial_weight_kg * 1.4)),
            carbs_g=float(maintenance * 0.5 / 4),
            fat_g=float(maintenance * 0.3 / 9),
            sleep_hours=profile.sleep_baseline_h,
            steps=profile.activity_baseline_steps,
            training_load=0.5,
            recovery_score=0.7,
            energy_score=0.7,
            adherence_score=0.8,
            day_index=0,
        )
        state.weight_history.append(state.current_weight_kg)
        return state

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------
    def step(
        self,
        state: DynamicState,
        profile: UserProfile,
        deltas: "Deltas",
        rng: np.random.Generator,
    ) -> DynamicState:
        """Apply requested target adjustments, then simulate one day."""

        # 1) Update target values (these have already been clipped by safety)
        state.calorie_target += deltas.calorie
        state.sleep_target_h += deltas.sleep
        state.step_target += deltas.steps
        state.protein_target_g += deltas.protein

        # 2) Adherence penalty when target jumps are large relative to capacity
        change_magnitude = (
            abs(deltas.calorie) / 100.0
            + abs(deltas.sleep) * 1.0
            + abs(deltas.steps) / 1000.0
            + abs(deltas.protein) / 30.0
        )
        adherence_drop = min(self.max_adherence_drop, 0.02 * change_magnitude)
        state.adherence_score = float(np.clip(
            state.adherence_score - adherence_drop + rng.normal(0.01, 0.015),
            0.0, 1.0,
        ))

        # 3) Realized day: noisy execution of the targets, weighted by adherence
        eff = 0.5 + 0.5 * state.adherence_score  # 0.5-1.0 multiplier
        state.actual_calorie_intake = float(np.clip(
            state.calorie_target + rng.normal(0, 80) * (2 - eff),
            900, 5000,
        ))
        state.protein_g = float(np.clip(
            state.protein_target_g + rng.normal(0, 8) * (2 - eff), 30, 300,
        ))
        # Carbs and fat balance to roughly fill remaining calories
        remaining_kcal = max(state.actual_calorie_intake - 4 * state.protein_g, 600.0)
        carb_fraction = float(np.clip(0.55 + rng.normal(0, 0.03), 0.35, 0.7))
        state.carbs_g = float(remaining_kcal * carb_fraction / 4.0)
        state.fat_g = float(remaining_kcal * (1 - carb_fraction) / 9.0)

        state.sleep_hours = float(np.clip(
            state.sleep_target_h + rng.normal(0, 0.4) * (2 - eff), 3.5, 11.0,
        ))
        state.steps = int(np.clip(
            state.step_target + rng.normal(0, 800) * (2 - eff), 0, 25000,
        ))

        # 4) Training load: rolling average of normalized step intensity
        load_today = float(np.clip(state.steps / 12000.0, 0, 1.2))
        state.training_load = float(0.7 * state.training_load + 0.3 * load_today)

        # 5) Recovery depends on sleep & load
        sleep_score = float(np.clip((state.sleep_hours - 5.0) / 3.5, 0, 1))
        load_penalty = max(0.0, state.training_load - 0.85) * 0.6
        state.recovery_score = float(np.clip(
            0.7 * sleep_score + 0.3 * (1 - load_penalty) + rng.normal(0, 0.03),
            0.0, 1.0,
        ))

        # 6) Energy depends on intake adequacy and recovery
        maintenance = profile.maintenance_calories()
        intake_ratio = state.actual_calorie_intake / max(maintenance, 1.0)
        energy_from_intake = 1 - min(1.0, abs(intake_ratio - 1.0) * 1.4)
        state.energy_score = float(np.clip(
            0.55 * state.recovery_score + 0.45 * energy_from_intake + rng.normal(0, 0.03),
            0.0, 1.0,
        ))

        # 7) Slow weight evolution from cumulative calorie balance
        balance = state.actual_calorie_intake - maintenance - 0.04 * state.steps
        dw = balance / self.KCAL_PER_KG  # kg change today (very small)
        state.current_weight_kg = float(np.clip(
            state.current_weight_kg + dw, 35.0, 200.0,
        ))
        state.weight_history.append(state.current_weight_kg)

        # 8) Tick the day counter
        state.day_index += 1
        return state


@dataclass
class Deltas:
    """Per-day adjustment requested by the agent (already safety-clipped)."""

    calorie: float = 0.0   # kcal
    sleep: float = 0.0     # hours
    steps: int = 0         # steps
    protein: float = 0.0   # grams
