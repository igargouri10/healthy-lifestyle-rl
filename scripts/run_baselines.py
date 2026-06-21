"""Evaluate the random and rule-based baselines across all simulated profiles.

Writes:
  results/metrics/baselines_summary.csv
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lifestyle_rl.agents.baselines import RandomPolicy, RuleBasedPolicy
from lifestyle_rl.envs.healthy_lifestyle_env import HealthyLifestyleEnv
from lifestyle_rl.simulation.user_profiles import PROFILE_LIBRARY
from lifestyle_rl.utils import set_global_seed, summarize_episode


def rollout(env: HealthyLifestyleEnv, policy, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    total = 0.0
    infos = []
    done = False
    while not done:
        a = policy.act(obs, info)
        obs, r, term, trunc, info = env.step(a)
        infos.append(info)
        total += r
        done = term or trunc
    return summarize_episode(infos, total).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=str,
        default="results/metrics/baselines_summary.csv",
    )
    args = parser.parse_args()
    set_global_seed(args.seed)

    rows = []
    for profile_name in PROFILE_LIBRARY:
        env = HealthyLifestyleEnv(profile_name=profile_name)
        for policy_name, policy in [
            ("random", RandomPolicy(seed=args.seed)),
            ("rule_based", RuleBasedPolicy()),
        ]:
            ep_rewards = []
            interventions = []
            adherence = []
            for ep in range(args.episodes):
                stats = rollout(env, policy, seed=args.seed + ep * 17)
                ep_rewards.append(stats["total_reward"])
                interventions.append(stats["safety_interventions"])
                adherence.append(stats["avg_adherence"])
            rows.append({
                "profile": profile_name,
                "policy": policy_name,
                "episodes": args.episodes,
                "mean_reward": float(np.mean(ep_rewards)),
                "std_reward": float(np.std(ep_rewards)),
                "mean_safety_interventions": float(np.mean(interventions)),
                "mean_adherence": float(np.mean(adherence)),
            })
            print(f"[{profile_name:<22}] {policy_name:<10}  "
                  f"reward={np.mean(ep_rewards):7.2f} +/- {np.std(ep_rewards):.2f}  "
                  f"interventions/ep={np.mean(interventions):4.1f}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved baseline summary -> {out}")


if __name__ == "__main__":
    main()
