"""Static user profile definitions.

These are *simulated* personas used purely for RL experimentation.
They are not medical recommendations and they do not target unhealthy
body composition outcomes.  The five profiles intentionally span the
goal categories called out in section 11 of the project prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


GOAL_TYPES: List[str] = [
    "maintenance",          # hold the user's current healthy steady state
    "general_wellness",     # gentle improvements across sleep/activity/nutrition
    "activity_consistency", # build a sustainable activity routine
    "gradual_management",   # very slow, healthy weight management
    "performance_support",  # support consistent training without overreach
]


@dataclass
class UserProfile:
    """A static simulated user. All values are configurable per profile."""

    name: str
    age: int
    sex_code: int          # 0 or 1; used only for the BMR formula
    height_cm: float
    initial_weight_kg: float
    goal_type: str
    activity_baseline_steps: int = 6000
    sleep_baseline_h: float = 7.5

    def basal_metabolic_rate(self) -> float:
        """Mifflin-St Jeor estimate (kcal/day). Educational simulation only."""
        s = 5 if self.sex_code == 1 else -161
        return 10 * self.initial_weight_kg + 6.25 * self.height_cm - 5 * self.age + s

    def maintenance_calories(self, activity_factor: float = 1.4) -> float:
        return self.basal_metabolic_rate() * activity_factor


PROFILE_LIBRARY: Dict[str, UserProfile] = {
    "maintenance": UserProfile(
        name="maintenance",
        age=34, sex_code=1, height_cm=178, initial_weight_kg=76.0,
        goal_type="maintenance",
        activity_baseline_steps=7000, sleep_baseline_h=7.5,
    ),
    "general_wellness": UserProfile(
        name="general_wellness",
        age=29, sex_code=0, height_cm=165, initial_weight_kg=64.0,
        goal_type="general_wellness",
        activity_baseline_steps=5500, sleep_baseline_h=7.0,
    ),
    "activity_consistency": UserProfile(
        name="activity_consistency",
        age=42, sex_code=1, height_cm=182, initial_weight_kg=85.0,
        goal_type="activity_consistency",
        activity_baseline_steps=4500, sleep_baseline_h=6.8,
    ),
    "gradual_management": UserProfile(
        name="gradual_management",
        age=47, sex_code=0, height_cm=170, initial_weight_kg=82.0,
        goal_type="gradual_management",
        activity_baseline_steps=5000, sleep_baseline_h=7.0,
    ),
    "performance_support": UserProfile(
        name="performance_support",
        age=26, sex_code=1, height_cm=176, initial_weight_kg=72.0,
        goal_type="performance_support",
        activity_baseline_steps=9000, sleep_baseline_h=8.0,
    ),
}


def sample_profile(
    rng: np.random.Generator,
    name: Optional[str] = None,
    jitter: bool = True,
) -> UserProfile:
    """Return a profile from the library, optionally with small noise.

    If ``name`` is given, that profile is returned (jittered).  Otherwise
    one is drawn uniformly from the library.
    """
    if name is None:
        name = rng.choice(list(PROFILE_LIBRARY.keys()))
    base = PROFILE_LIBRARY[name]
    if not jitter:
        return base
    return UserProfile(
        name=base.name,
        age=int(np.clip(base.age + rng.integers(-3, 4), 18, 70)),
        sex_code=base.sex_code,
        height_cm=float(np.clip(base.height_cm + rng.normal(0, 2.0), 150, 200)),
        initial_weight_kg=float(np.clip(base.initial_weight_kg + rng.normal(0, 2.5), 45, 130)),
        goal_type=base.goal_type,
        activity_baseline_steps=int(np.clip(base.activity_baseline_steps + rng.integers(-700, 700), 2000, 14000)),
        sleep_baseline_h=float(np.clip(base.sleep_baseline_h + rng.normal(0, 0.3), 5.5, 9.0)),
    )
