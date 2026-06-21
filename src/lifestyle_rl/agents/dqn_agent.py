"""A minimal but complete DQN agent written from scratch in PyTorch.

Includes:

  * an MLP Q-network and a target network
  * an experience replay buffer (see ``replay_buffer.py``)
  * epsilon-greedy exploration with linear decay
  * the standard Bellman target
  * periodic hard target-network sync
  * train / save / load helpers

The training loop lives in ``scripts/train_dqn.py``; this module is the
algorithm itself, kept library-style for easy reuse from tests and from
``tune_bayesian.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    obs_dim: int = 18
    n_actions: int = 8
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 50000
    min_replay_size: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 200
    target_update_freq: int = 500
    train_freq: int = 4
    grad_clip: float = 10.0
    device: str = "cpu"
    seed: int = 0


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.q = QNetwork(cfg.obs_dim, cfg.n_actions, cfg.hidden_sizes).to(self.device)
        self.target = QNetwork(cfg.obs_dim, cfg.n_actions, cfg.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=cfg.learning_rate)
        self.buffer = ReplayBuffer(cfg.buffer_size, cfg.obs_dim, seed=cfg.seed)
        self._step_count = 0
        self._train_step_count = 0
        self._losses: List[float] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def epsilon(self, episode: int) -> float:
        cfg = self.cfg
        if cfg.epsilon_decay_episodes <= 0:
            return cfg.epsilon_end
        frac = min(1.0, episode / cfg.epsilon_decay_episodes)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def act(self, obs: np.ndarray, epsilon: float = 0.0, rng: Optional[np.random.Generator] = None) -> int:
        rng = rng or np.random.default_rng()
        if rng.random() < epsilon:
            return int(rng.integers(0, self.cfg.n_actions))
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(t)
            return int(torch.argmax(q, dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def remember(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)
        self._step_count += 1

    def train_step(self) -> Optional[float]:
        cfg = self.cfg
        if len(self.buffer) < cfg.min_replay_size:
            return None
        if self._step_count % cfg.train_freq != 0:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(cfg.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        nobs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        r_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        d_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q(s,a)
        q_sa = self.q(obs_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

        # Bellman target with target network
        with torch.no_grad():
            q_next = self.target(nobs_t).max(dim=1).values
            target = r_t + cfg.gamma * (1 - d_t) * q_next

        loss = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip)
        self.optim.step()

        self._train_step_count += 1
        if self._step_count % cfg.target_update_freq == 0:
            self.target.load_state_dict(self.q.state_dict())

        loss_val = float(loss.item())
        self._losses.append(loss_val)
        return loss_val

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q": self.q.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.target.load_state_dict(ckpt["q"])

    @property
    def recent_loss(self) -> float:
        return float(np.mean(self._losses[-100:])) if self._losses else 0.0
