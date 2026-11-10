from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.helpers import SinusoidalPosEmb, init_weights
from agent.topology_repair import repair_scores_to_radial_topology


class CategoricalTopologyModel(nn.Module):
    """离散拓扑扩散的去噪网络。

    输入是状态、当前离散噪声拓扑 x_t 和时间步 t；输出每条支路 open/closed
    的分类 logits。这里仍使用 MLP，是为了和现有连续 diffusion 的网络容量保持接近。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256, time_dim: int = 32):
        super().__init__()
        self.action_dim = int(action_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, time_dim),
        )
        input_dim = state_dim + action_dim * 2 + time_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, action_dim * 2),
        )
        self.apply(init_weights)

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if x_t.dim() != 2:
            raise ValueError(f"x_t must have shape [batch, action_dim], got {tuple(x_t.shape)}")
        x_one_hot = F.one_hot(x_t.long(), num_classes=2).float().view(x_t.shape[0], -1)
        t_emb = self.time_mlp(time)
        logits = self.layer(torch.cat([state, x_one_hot, t_emb], dim=-1))
        return logits.view(x_t.shape[0], self.action_dim, 2)


class CategoricalDiffusion(nn.Module):
    """B1 版本：Categorical diffusion + 每步 radial projection。

    训练时随机腐蚀最优拓扑 x0 得到 x_t，并学习还原 x0。采样时每个反向步骤
    输出支路闭合 logits，用 Gumbel 扰动保持候选多样性，再用 Kruskal repair
    投影到径向可行拓扑。这样约束进入扩散链内部，而不是只在最终输出后修复。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_timesteps: int,
        branch_data: Sequence[Dict[str, float]],
        num_buses: int,
        max_corruption: float = 0.5,
        sample_temperature: float = 1.0,
        hidden_size: int = 256,
        behavior_sample: int = 16,
        eval_sample: int = 512,
        deterministic: bool = False,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_timesteps = int(n_timesteps)
        self.branch_data = list(branch_data)
        self.num_buses = int(num_buses)
        self.max_corruption = float(max_corruption)
        self.sample_temperature = float(sample_temperature)
        self.behavior_sample = int(behavior_sample)
        self.eval_sample = int(eval_sample)
        self.deterministic = bool(deterministic)
        self.model = CategoricalTopologyModel(state_dim, action_dim, hidden_size=hidden_size)

        if self.n_timesteps <= 0:
            raise ValueError(f"n_timesteps must be positive, got {n_timesteps}")
        if not 0.0 <= self.max_corruption <= 1.0:
            raise ValueError(f"max_corruption must be in [0, 1], got {max_corruption}")
        if self.sample_temperature <= 0.0:
            raise ValueError(f"sample_temperature must be positive, got {sample_temperature}")

    @staticmethod
    def _action_to_binary(action: torch.Tensor) -> torch.Tensor:
        if action.dim() != 2:
            raise ValueError(f"action must have shape [batch, action_dim], got {tuple(action.shape)}")
        return (action > 0.0).long()

    @staticmethod
    def _binary_to_action(binary_topology: torch.Tensor) -> torch.Tensor:
        return binary_topology.float() * 2.0 - 1.0

    def _corruption_probability(self, t: torch.Tensor) -> torch.Tensor:
        return self.max_corruption * (t.float() + 1.0) / float(self.n_timesteps)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_start.dim() != 2:
            raise ValueError(f"x_start must have shape [batch, action_dim], got {tuple(x_start.shape)}")
        corruption_prob = self._corruption_probability(t).view(-1, 1)
        random_bits = torch.randint(0, 2, x_start.shape, device=x_start.device, dtype=torch.long)
        replace_mask = torch.rand(x_start.shape, device=x_start.device) < corruption_prob
        return torch.where(replace_mask, random_bits, x_start.long())

    def loss(self, action: torch.Tensor, state: torch.Tensor, weights=1.0) -> torch.Tensor:
        del weights
        x_start = self._action_to_binary(action)
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=action.device).long()
        x_noisy = self.q_sample(x_start, t)
        logits = self.model(x_noisy, t, state)
        return F.cross_entropy(logits.view(-1, 2), x_start.view(-1))

    def _initial_topology(self, batch_size: int, device: torch.device) -> torch.Tensor:
        initial_scores = torch.rand(batch_size, self.action_dim, device=device)
        return self._project_scores(initial_scores)

    def _project_scores(self, closed_scores: torch.Tensor) -> torch.Tensor:
        if closed_scores.dim() != 2:
            raise ValueError(f"closed_scores must have shape [batch, action_dim], got {tuple(closed_scores.shape)}")
        projected = []
        scores_np = closed_scores.detach().cpu().numpy()
        for row in scores_np:
            repaired = repair_scores_to_radial_topology(row, self.branch_data, self.num_buses)
            projected.append(repaired.topology)
        return torch.as_tensor(np.asarray(projected, dtype=np.float32), device=closed_scores.device).long()

    def _gumbel_noise(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        uniform = torch.rand(shape, device=device).clamp_(1e-6, 1.0 - 1e-6)
        return -torch.log(-torch.log(uniform))

    def _project_logits(self, logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
        closed_logits = logits[:, :, 1]
        if deterministic:
            scores = closed_logits
        else:
            scores = closed_logits + self._gumbel_noise(closed_logits.shape, closed_logits.device) * self.sample_temperature
        return self._project_scores(scores)

    @torch.no_grad()
    def p_sample_loop(self, state: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        if len(shape) != 2:
            raise ValueError(f"shape must be [batch, action_dim], got {shape}")
        batch_size, action_dim = int(shape[0]), int(shape[1])
        if action_dim != self.action_dim:
            raise ValueError(f"shape action_dim must be {self.action_dim}, got {action_dim}")
        if state.shape[0] != batch_size:
            raise ValueError(f"state batch {state.shape[0]} does not match sample batch {batch_size}")

        x_t = self._initial_topology(batch_size, state.device)
        use_deterministic = self.deterministic
        for step in reversed(range(self.n_timesteps)):
            t = torch.full((batch_size,), step, device=state.device, dtype=torch.long)
            logits = self.model(x_t, t, state)
            x_t = self._project_logits(logits, deterministic=use_deterministic)
        return self._binary_to_action(x_t)

    @torch.no_grad()
    def sample(self, state: torch.Tensor, eval: bool = False, q_func=None, normal: bool = False) -> torch.Tensor:
        del q_func
        del normal
        batch_size = state.shape[0]
        return self.p_sample_loop(state, (batch_size, self.action_dim))

    def forward(self, state: torch.Tensor, eval: bool = False, q_func=None, normal: bool = False) -> torch.Tensor:
        return self.sample(state, eval=eval, q_func=q_func, normal=normal)
