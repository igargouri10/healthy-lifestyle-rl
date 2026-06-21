"""Two non-learning baselines used as comparison points for DQN and PPO.

Both implement the simple interface::

    policy.act(observation_vec, info_dict) -> int

The ``info_dict`` is the same dictionary returned by
``HealthyLifestyleEnv.step()`` and lets the rule-based policy peek at
adherence/recovery directly without having to decode the observation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..safety.safety_layer import N_ACTIONS


class RandomPolicy:
    """Uniform random over the discrete action space."""

    def __init__(self, n_actions: int = N_ACTIONS, seed: Optional[int] = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> int:
        return int(self.rng.integers(0, self.n_actions))


class RuleBasedPolicy:
    """A simple heuristic fitness-planner policy.

    Logic follows section 9.2 of the project prompt:

      * if adherence is low: maintain (don't push more changes)
      * if recovery is low: recommend recovery (decrease activity)
      * if sleep is low: recommend more sleep
      * if goal progress is slow and recovery is acceptable: nudge calories
      * otherwise maintain
    """

    def __init__(
        self,
        adherence_floor: float = 0.55,
        recovery_floor: float = 0.5,
        sleep_floor_h: float = 7.0,
    ):
        self.adherence_floor = adherence_floor
        self.recovery_floor = recovery_floor
        self.sleep_floor_h = sleep_floor_h

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> int:
        info = info or {}
        adherence = info.get("adherence_score", 1.0)
        recovery = info.get("recovery_score", 1.0)
        sleep_h = info.get("sleep_hours", 7.5)
        trend = info.get("weight_trend_7d", 0.0)

        # 1) adherence too low? maintain
        if adherence < self.adherence_floor:
            return 0

        # 2) recovery too low? recover
        if recovery < self.recovery_floor:
            return 4

        # 3) sleep too low? boost sleep
        if sleep_h < self.sleep_floor_h:
            return 5

        # 4) goal progress nudges
        # Slight calorie cut if trending strongly upward; slight bump if dropping fast
        if trend > 0.3:
            return 1
        if trend < -0.3:
            return 2

        # 5) default: maintain
        return 0
