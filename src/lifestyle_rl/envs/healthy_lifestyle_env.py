"""HealthyLifestyleEnv: a Gymnasium-compatible RL environment.

The agent observes an 18-dimensional normalized state vector and picks one of
8 discrete adjustment actions.  The safety layer translates the action into
clipped per-day target deltas; the transition model simulates one day; the
reward function decomposes the daily outcome into 8 weighted components.

This is *simulation only*. It does not give medical advice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from ..rewards.reward_functions import RewardConfig, compute_reward
from ..safety.safety_layer import N_ACTIONS, SafetyConfig, SafetyLayer, ACTION_MEANINGS
from ..simulation.transition_model import DynamicState, TransitionModel
from ..simulation.user_profiles import PROFILE_LIBRARY, UserProfile, sample_profile


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "env.yaml"


def _load_default_config() -> Dict[str, Any]:
    """Load the YAML config if present; fall back to safe defaults otherwise."""
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {
        "env": {"episode_days": 60, "seed": 0, "default_profile": "general_wellness"},
        "safety": {},
        "rewards": {},
    }


class HealthyLifestyleEnv(gym.Env):
    """A Gymnasium env modeling daily lifestyle planning as an MDP."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    OBS_KEYS = [
        "age_norm", "sex_code", "height_norm", "initial_weight_norm",
        "goal_code",
        "current_weight_norm", "weight_trend_7d_norm",
        "calorie_target_norm", "actual_calorie_intake_norm",
        "protein_norm", "carbs_norm", "fat_norm",
        "sleep_hours_norm", "steps_norm",
        "recovery_score", "adherence_score", "energy_score",
        "day_index_norm",
    ]
    OBS_DIM = len(OBS_KEYS)

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        profile_name: Optional[str] = None,
        jitter_profile: bool = True,
    ):
        super().__init__()
        cfg = _load_default_config() if config is None else config
        env_cfg = cfg.get("env", {})
        self._safety_cfg = SafetyConfig.from_dict(cfg.get("safety", {}) or {})
        self._reward_cfg = RewardConfig.from_dict(cfg.get("rewards", {}) or {})

        self.episode_days: int = int(env_cfg.get("episode_days", 60))
        self._profile_name = profile_name or env_cfg.get("default_profile", "general_wellness")
        self._jitter_profile = bool(jitter_profile)
        self._initial_seed = int(env_cfg.get("seed", 0))
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.OBS_DIM,), dtype=np.float32,
        )

        self.safety_layer = SafetyLayer(self._safety_cfg)
        self.transition = TransitionModel()

        self._rng: np.random.Generator = np.random.default_rng(self._initial_seed)
        self.profile: UserProfile = PROFILE_LIBRARY[self._profile_name]
        self.state: DynamicState = self.transition.init_state(self.profile, self._rng)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        name = options.get("profile_name", self._profile_name)
        jitter = options.get("jitter_profile", self._jitter_profile)
        self.profile = sample_profile(self._rng, name=name, jitter=jitter)
        self.state = self.transition.init_state(self.profile, self._rng)
        return self._observation(), self._build_info(action=0, safety_info={}, components={})

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; must be in {self.action_space}")

        deltas, safety_info = self.safety_layer(self.state, action, self.state.current_weight_kg)
        self.state = self.transition.step(self.state, self.profile, deltas, self._rng)
        reward, components = compute_reward(
            self.state, self.profile, deltas, safety_info, self._reward_cfg,
        )

        terminated = False
        truncated = self.state.day_index >= self.episode_days

        info = self._build_info(action=action, safety_info=safety_info, components=components)
        info["reward"] = reward

        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> None:
        s = self.state
        print(
            f"Day {s.day_index:>2d} | "
            f"weight={s.current_weight_kg:5.2f}kg trend7d={s.weight_trend_7d:+.2f} | "
            f"cal_t={s.calorie_target:5.0f} sleep_t={s.sleep_target_h:.2f}h "
            f"steps_t={s.step_target:>5d} prot_t={s.protein_target_g:5.1f}g | "
            f"adh={s.adherence_score:.2f} rec={s.recovery_score:.2f} en={s.energy_score:.2f}"
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        p, s = self.profile, self.state
        goal_idx = {
            "maintenance": 0, "general_wellness": 1, "activity_consistency": 2,
            "gradual_management": 3, "performance_support": 4,
        }.get(p.goal_type, 1)

        vec = np.array([
            (p.age - 35) / 20.0,
            float(p.sex_code),
            (p.height_cm - 170) / 20.0,
            (p.initial_weight_kg - 75) / 25.0,
            goal_idx / 4.0,
            (s.current_weight_kg - p.initial_weight_kg) / 10.0,
            np.clip(s.weight_trend_7d / 1.0, -2.0, 2.0),
            (s.calorie_target - 2200) / 800.0,
            (s.actual_calorie_intake - 2200) / 800.0,
            (s.protein_g - 100) / 60.0,
            (s.carbs_g - 250) / 150.0,
            (s.fat_g - 70) / 40.0,
            (s.sleep_hours - 7.5) / 1.5,
            (s.steps - 7000) / 4000.0,
            s.recovery_score,
            s.adherence_score,
            s.energy_score,
            s.day_index / max(self.episode_days, 1),
        ], dtype=np.float32)
        return np.clip(vec, -2.0, 2.0)

    def _build_info(self, action: int, safety_info: Dict, components: Dict) -> Dict[str, Any]:
        s = self.state
        info = {
            "raw_action": safety_info.get("raw_action", action),
            "safe_action": safety_info.get("safe_action", action),
            "safety_intervention_applied": safety_info.get("safety_intervention_applied", False),
            "unsafe_proposal": safety_info.get("unsafe_proposal", False),
            "clipped_components": safety_info.get("clipped_components", []),
            "action_meaning": ACTION_MEANINGS.get(action, "?"),
            "current_weight_kg": s.current_weight_kg,
            "weight_trend_7d": s.weight_trend_7d,
            "adherence_score": s.adherence_score,
            "recovery_score": s.recovery_score,
            "energy_score": s.energy_score,
            "calorie_target": s.calorie_target,
            "sleep_hours": s.sleep_hours,
            "steps": s.steps,
            "reward_components": components,
        }
        return info
