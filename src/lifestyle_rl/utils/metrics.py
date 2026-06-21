"""Metric containers used by training, evaluation, and tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EpisodeStats:
    """Per-episode rollup of environment information."""

    total_reward: float = 0.0
    steps: int = 0
    safety_interventions: int = 0
    unsafe_proposals: int = 0
    avg_adherence: float = 0.0
    avg_recovery: float = 0.0
    avg_sleep_hours: float = 0.0
    avg_steps_per_day: float = 0.0
    weight_trend_final: float = 0.0
    action_counts: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        out = {
            "total_reward": float(self.total_reward),
            "steps": int(self.steps),
            "safety_interventions": int(self.safety_interventions),
            "unsafe_proposals": int(self.unsafe_proposals),
            "avg_adherence": float(self.avg_adherence),
            "avg_recovery": float(self.avg_recovery),
            "avg_sleep_hours": float(self.avg_sleep_hours),
            "avg_steps_per_day": float(self.avg_steps_per_day),
            "weight_trend_final": float(self.weight_trend_final),
        }
        for a, c in self.action_counts.items():
            out[f"action_{a}_count"] = int(c)
        return out


def summarize_episode(infos: List[Dict], total_reward: float) -> EpisodeStats:
    """Reduce a list of per-step `info` dicts into an EpisodeStats object."""
    if not infos:
        return EpisodeStats(total_reward=total_reward)

    n = len(infos)
    adherence = sum(i.get("adherence_score", 0.0) for i in infos) / n
    recovery = sum(i.get("recovery_score", 0.0) for i in infos) / n
    sleep = sum(i.get("sleep_hours", 0.0) for i in infos) / n
    steps_pd = sum(i.get("steps", 0.0) for i in infos) / n
    interventions = sum(1 for i in infos if i.get("safety_intervention_applied"))
    unsafe = sum(1 for i in infos if i.get("unsafe_proposal"))
    trend_final = infos[-1].get("weight_trend_7d", 0.0)

    counts: Dict[int, int] = {}
    for i in infos:
        a = int(i.get("safe_action", -1))
        counts[a] = counts.get(a, 0) + 1

    return EpisodeStats(
        total_reward=total_reward,
        steps=n,
        safety_interventions=interventions,
        unsafe_proposals=unsafe,
        avg_adherence=adherence,
        avg_recovery=recovery,
        avg_sleep_hours=sleep,
        avg_steps_per_day=steps_pd,
        weight_trend_final=trend_final,
        action_counts=counts,
    )
