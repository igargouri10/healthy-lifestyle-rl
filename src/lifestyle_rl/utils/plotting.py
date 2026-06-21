"""Matplotlib helpers for training and evaluation plots."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_training_curve(rewards: Sequence[float], out_path: str, title: str) -> None:
    """Episode-reward learning curve with a 10-episode moving average."""
    _ensure_parent(out_path)
    rewards = list(rewards)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rewards, alpha=0.4, label="Episode reward")
    if len(rewards) >= 10:
        window = 10
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), ma, label=f"{window}-ep MA", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_bar_comparison(
    labels: Sequence[str],
    values: Sequence[float],
    out_path: str,
    title: str,
    ylabel: str,
) -> None:
    _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#888", "#5B9BD5", "#70AD47", "#ED7D31", "#A5A5A5"][: len(labels)])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_action_distribution(
    counts_per_policy: Dict[str, Dict[int, int]],
    n_actions: int,
    out_path: str,
    title: str,
) -> None:
    """Grouped bar chart of action usage across policies."""
    _ensure_parent(out_path)
    policies = list(counts_per_policy.keys())
    x = np.arange(n_actions)
    width = 0.8 / max(len(policies), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, p in enumerate(policies):
        c = counts_per_policy[p]
        total = max(sum(c.values()), 1)
        freqs = [c.get(a, 0) / total for a in range(n_actions)]
        ax.bar(x + i * width - 0.4 + width / 2, freqs, width, label=p)
    ax.set_xticks(x)
    ax.set_xticklabels([f"a{a}" for a in range(n_actions)])
    ax.set_ylabel("Action frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_optuna_history(values: Iterable[float], out_path: str) -> None:
    _ensure_parent(out_path)
    values = list(values)
    if not values:
        return
    best = np.maximum.accumulate(values)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(values, "o-", alpha=0.5, label="Trial value")
    ax.plot(best, "-", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective (mean reward)")
    ax.set_title("Bayesian (Optuna) tuning history")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
