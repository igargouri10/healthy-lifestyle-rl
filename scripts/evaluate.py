"""Evaluate all policies across all simulated profiles.

Compares: random, rule-based, DQN, PPO.

Outputs:
  results/metrics/evaluation_summary.csv
  results/plots/reward_comparison.png
  results/plots/safety_intervention_comparison.png
  results/plots/adherence_comparison.png
  results/plots/action_distribution.png
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from lifestyle_rl.agents.baselines import RandomPolicy, RuleBasedPolicy
from lifestyle_rl.envs.healthy_lifestyle_env import HealthyLifestyleEnv
from lifestyle_rl.simulation.user_profiles import PROFILE_LIBRARY
from lifestyle_rl.safety.safety_layer import N_ACTIONS
from lifestyle_rl.utils import set_global_seed, summarize_episode
from lifestyle_rl.utils.plotting import (
    plot_action_distribution, plot_bar_comparison,
)


def _try_load_dqn(path: str):
    if not Path(path).exists():
        return None
    try:
        from lifestyle_rl.agents.dqn_agent import DQNAgent, DQNConfig
    except ImportError as e:
        print(f"Warning: torch not available, skipping DQN ({e})")
        return None
    agent = DQNAgent(DQNConfig())
    try:
        agent.load(path)
        return agent
    except Exception as e:
        print(f"Warning: could not load DQN model at {path}: {e}")
        return None


def _try_load_ppo(path: str):
    if not Path(path).exists():
        return None
    try:
        from stable_baselines3 import PPO
        return PPO.load(path)
    except Exception as e:
        print(f"Warning: could not load PPO model at {path}: {e}")
        return None


def make_act_fn(name: str, dqn, ppo):
    rng = np.random.default_rng(0)
    random_p = RandomPolicy(seed=0)
    rule_p = RuleBasedPolicy()

    def act(obs, info):
        if name == "random":
            return random_p.act(obs, info)
        if name == "rule_based":
            return rule_p.act(obs, info)
        if name == "dqn" and dqn is not None:
            return dqn.act(obs, epsilon=0.0, rng=rng)
        if name == "ppo" and ppo is not None:
            a, _ = ppo.predict(obs, deterministic=True)
            return int(a)
        # Fallback if model isn't available
        return 0
    return act


def rollout(env: HealthyLifestyleEnv, act_fn, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    total = 0.0
    infos = []
    done = False
    while not done:
        a = act_fn(obs, info)
        obs, r, term, trunc, info = env.step(a)
        infos.append(info)
        total += r
        done = term or trunc
    return summarize_episode(infos, total).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dqn-path", type=str, default="results/trained_models/dqn_model.pt")
    parser.add_argument("--ppo-path", type=str, default="results/trained_models/ppo_model.zip")
    parser.add_argument("--out-csv", type=str,
                        default="results/metrics/evaluation_summary.csv")
    args = parser.parse_args()
    set_global_seed(args.seed)

    dqn = _try_load_dqn(args.dqn_path)
    ppo = _try_load_ppo(args.ppo_path)

    policy_names: List[str] = ["random", "rule_based"]
    if dqn is not None: policy_names.append("dqn")
    if ppo is not None: policy_names.append("ppo")

    rows = []
    action_counts: Dict[str, Dict[int, int]] = {p: {a: 0 for a in range(N_ACTIONS)} for p in policy_names}
    for profile in PROFILE_LIBRARY:
        env = HealthyLifestyleEnv(profile_name=profile)
        for policy in policy_names:
            act_fn = make_act_fn(policy, dqn, ppo)
            ep_rewards, ep_interventions, ep_adherence, ep_recovery = [], [], [], []
            for ep in range(args.episodes):
                stats = rollout(env, act_fn, seed=args.seed + ep * 31)
                ep_rewards.append(stats["total_reward"])
                ep_interventions.append(stats["safety_interventions"])
                ep_adherence.append(stats["avg_adherence"])
                ep_recovery.append(stats["avg_recovery"])
                for a in range(N_ACTIONS):
                    action_counts[policy][a] += stats.get(f"action_{a}_count", 0)
            rows.append({
                "profile": profile, "policy": policy, "episodes": args.episodes,
                "mean_reward": float(np.mean(ep_rewards)),
                "std_reward": float(np.std(ep_rewards)),
                "mean_safety_interventions": float(np.mean(ep_interventions)),
                "mean_adherence": float(np.mean(ep_adherence)),
                "mean_recovery": float(np.mean(ep_recovery)),
            })
            print(f"[{profile:<22}] {policy:<10} reward={np.mean(ep_rewards):7.2f}  "
                  f"interventions={np.mean(ep_interventions):5.1f}  "
                  f"adherence={np.mean(ep_adherence):.2f}")

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved -> {args.out_csv}")

    # Aggregate plots across profiles
    agg = df.groupby("policy", as_index=False).agg(
        mean_reward=("mean_reward", "mean"),
        mean_safety_interventions=("mean_safety_interventions", "mean"),
        mean_adherence=("mean_adherence", "mean"),
    )
    plot_bar_comparison(
        agg["policy"].tolist(), agg["mean_reward"].tolist(),
        "results/plots/reward_comparison.png",
        "Mean episode reward (avg over profiles)", "Reward",
    )
    plot_bar_comparison(
        agg["policy"].tolist(), agg["mean_safety_interventions"].tolist(),
        "results/plots/safety_intervention_comparison.png",
        "Mean safety interventions / episode", "Interventions",
    )
    plot_bar_comparison(
        agg["policy"].tolist(), agg["mean_adherence"].tolist(),
        "results/plots/adherence_comparison.png",
        "Mean adherence score (0-1)", "Adherence",
    )
    plot_action_distribution(
        action_counts, N_ACTIONS,
        "results/plots/action_distribution.png",
        "Action distribution by policy (all profiles aggregated)",
    )
    print("Saved comparison plots under results/plots/")


if __name__ == "__main__":
    main()
