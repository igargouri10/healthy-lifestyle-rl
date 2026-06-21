"""Visual inspection of a single episode for one profile across all policies.

Usage:
    python scripts/inspect_episode.py --profile gradual_management
    python scripts/inspect_episode.py --profile maintenance --episodes 3
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from lifestyle_rl.envs.healthy_lifestyle_env import HealthyLifestyleEnv
from lifestyle_rl.agents.baselines import RandomPolicy, RuleBasedPolicy
from lifestyle_rl.agents.dqn_agent import DQNAgent, DQNConfig
from lifestyle_rl.safety.safety_layer import ACTION_MEANINGS


ACTION_LABELS = {
    0: "maintain",
    1: "↓cal",
    2: "↑cal",
    3: "↑steps",
    4: "↓steps/rec",
    5: "↑sleep",
    6: "↑protein",
    7: "rebal-carbs",
}

POLICY_COLORS = {
    "random":     "#d62728",
    "rule_based": "#2ca02c",
    "dqn":        "#1f77b4",
    "ppo":        "#ff7f0e",
}


def rollout(policy_name: str, policy, env: HealthyLifestyleEnv, seed: int):
    """Run one episode and return per-day records."""
    obs, info = env.reset(seed=seed)
    days: List[Dict] = []
    done = False
    while not done:
        if policy_name in ("random", "rule_based"):
            action = policy.act(obs, info)
        elif policy_name == "dqn":
            action = policy.act(obs, epsilon=0.0)
        else:  # ppo
            action, _ = policy.predict(obs, deterministic=True)
            action = int(action)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        days.append({
            "day": info.get("day", env.state.day_index),
            "action": action,
            "intervention": int(info.get("safety_intervention_applied", False)),
            "weight": info.get("current_weight_kg"),
            "trend": info.get("weight_trend_7d"),
            "adherence": info.get("adherence_score"),
            "recovery": info.get("recovery_score"),
            "sleep": info.get("sleep_hours"),
            "steps": info.get("steps"),
            "reward": reward,
            "calorie": info.get("calorie_target"),
        })
    return days


def plot_inspection(all_days: Dict[str, List[Dict]], profile: str, out_path: str) -> None:
    policies = list(all_days.keys())
    n_days = len(next(iter(all_days.values())))
    days_x = list(range(1, n_days + 1))

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle(f"Episode inspection — profile: {profile}", fontsize=13, fontweight="bold")

    # --- Panel 1: cumulative reward ------------------------------------------
    ax = axes[0]
    for pol in policies:
        rewards = [d["reward"] for d in all_days[pol]]
        cumulative = np.cumsum(rewards)
        ax.plot(days_x, cumulative, label=pol, color=POLICY_COLORS.get(pol, "gray"), linewidth=1.8)
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative reward over the episode")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 2: safety interventions (raster) --------------------------------
    ax = axes[1]
    for i, pol in enumerate(policies):
        ivs = [d["day"] for d in all_days[pol] if d["intervention"]]
        ax.scatter(ivs, [i] * len(ivs), marker="|", s=200,
                   color=POLICY_COLORS.get(pol, "gray"), linewidths=2)
        ax.text(n_days + 0.5, i, f"{len(ivs)}", va="center", fontsize=8)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies)
    ax.set_title("Safety interventions per day (each tick = 1 clip)")
    ax.grid(axis="x", alpha=0.2)

    # --- Panel 3: weight trend 7d ---------------------------------------------
    ax = axes[2]
    for pol in policies:
        trend = [d["trend"] for d in all_days[pol]]
        ax.plot(days_x, trend, label=pol, color=POLICY_COLORS.get(pol, "gray"), linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel("kg / 7-day window")
    ax.set_title("7-day weight trend  (negative = losing weight)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 4: adherence and recovery --------------------------------------
    ax = axes[3]
    for pol in policies:
        adh = [d["adherence"] for d in all_days[pol]]
        rec = [d["recovery"] for d in all_days[pol]]
        c = POLICY_COLORS.get(pol, "gray")
        ax.plot(days_x, adh, color=c, linewidth=1.5, label=f"{pol} adh")
        ax.plot(days_x, rec, color=c, linewidth=1.0, linestyle="--", alpha=0.6, label=f"{pol} rec")
    ax.set_ylabel("Score [0-1]")
    ax.set_title("Adherence (solid) and Recovery (dashed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # --- Panel 5: action heatmap per policy -----------------------------------
    ax = axes[4]
    n_pol = len(policies)
    n_actions = 8
    heatmap = np.zeros((n_pol, n_days), dtype=int)
    for i, pol in enumerate(policies):
        for j, d in enumerate(all_days[pol]):
            heatmap[i, j] = d["action"]
    im = ax.imshow(heatmap, aspect="auto", cmap="tab10", vmin=0, vmax=7,
                   extent=[0.5, n_days + 0.5, -0.5, n_pol - 0.5])
    ax.set_yticks(range(n_pol))
    ax.set_yticklabels(policies)
    ax.set_xlabel("Day")
    ax.set_title("Action chosen each day  (color = action id 0-7)")
    patches = [mpatches.Patch(color=plt.cm.tab10(a / 9), label=f"{a}: {ACTION_LABELS[a]}")
               for a in range(n_actions)]
    ax.legend(handles=patches, fontsize=7, loc="upper right", ncol=2,
              bbox_to_anchor=(1.0, -0.25))

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved inspection plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="gradual_management",
                        help="Profile name to inspect")
    parser.add_argument("--seed", type=int, default=7,
                        help="Episode seed")
    parser.add_argument("--dqn-model", type=str,
                        default="results/trained_models/dqn_model.pt")
    parser.add_argument("--ppo-model", type=str,
                        default="results/trained_models/ppo_model.zip")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_path = args.out or f"results/plots/inspect_{args.profile}_ep{args.seed}.png"

    env_kwargs = dict(profile_name=args.profile, jitter_profile=False)

    # Build each policy
    policies = {}
    policies["random"] = RandomPolicy(seed=args.seed)
    policies["rule_based"] = RuleBasedPolicy()

    try:
        import torch
        dqn_cfg = DQNConfig()
        agent = DQNAgent(dqn_cfg)
        agent.load(args.dqn_model)
        policies["dqn"] = agent
    except Exception as e:
        print(f"[warn] Could not load DQN: {e}")

    try:
        from stable_baselines3 import PPO
        policies["ppo"] = PPO.load(args.ppo_model)
    except Exception as e:
        print(f"[warn] Could not load PPO: {e}")

    # Run one episode per policy
    all_days: Dict[str, List] = {}
    for pol_name, pol in policies.items():
        env = HealthyLifestyleEnv(**env_kwargs)
        days = rollout(pol_name, pol, env, seed=args.seed)
        all_days[pol_name] = days
        n_iv = sum(d["intervention"] for d in days)
        total_r = sum(d["reward"] for d in days)
        print(f"  {pol_name:12s}  reward={total_r:7.1f}  interventions={n_iv:3d}  "
              f"final_weight={days[-1]['weight']:.2f}kg  "
              f"final_trend={days[-1]['trend']:+.3f} kg/7d")

    plot_inspection(all_days, args.profile, out_path)


if __name__ == "__main__":
    main()
