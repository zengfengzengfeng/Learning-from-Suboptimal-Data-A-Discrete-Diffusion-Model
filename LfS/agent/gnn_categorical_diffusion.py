from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.gnn_model import TopologyGNNEncoder, build_edge_index, build_static_edge_features
from agent.helpers import SinusoidalPosEmb, init_weights
from agent.topology_repair import repair_scores_to_radial_topology


class GNNCategoricalTopologyModel(nn.Module):
    """GNN 版 categorical diffusion 去噪器。

    MLP 版本把整张网络压成一个扁平向量；这里把 state 还原为节点特征，把 x_t 作为支路开关特征，
    让每条支路的 open/closed logits 由相邻节点和支路参数共同决定。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        branch_data: Sequence[Dict[str, float]],
        num_buses: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        time_dim: int = 32,
    ):
        super().__init__()
        if state_dim % num_buses != 0:
            raise ValueError(f"state_dim={state_dim} cannot be reshaped to num_buses={num_buses}")
        if action_dim != len(branch_data):
            raise ValueError(f"action_dim must match branch count {len(branch_data)}, got {action_dim}")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.num_buses = int(num_buses)
        self.node_feature_dim = int(state_dim) // int(num_buses)
        self.time_dim = int(time_dim)
        self.register_buffer("edge_index", build_edge_index(branch_data, num_buses))
        self.register_buffer("static_edge_features", build_static_edge_features(branch_data))
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, time_dim),
        )
        self.encoder = TopologyGNNEncoder(
            node_input_dim=self.node_feature_dim + time_dim,
            edge_input_dim=4 + time_dim,
            hidden_dim=hidden_size,
            num_layers=num_layers,
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, 2),
        )
        self.apply(init_weights)

    def _node_features(self, state: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        if state.dim() != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape [batch, {self.state_dim}], got {tuple(state.shape)}")
        node_state = state.view(state.shape[0], self.node_feature_dim, self.num_buses).transpose(1, 2).contiguous()
        node_time = time_embedding.unsqueeze(1).expand(-1, self.num_buses, -1)
        return torch.cat([node_state, node_time], dim=-1)

    def _edge_features(self, x_t: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        if x_t.dim() != 2 or x_t.shape[1] != self.action_dim:
            raise ValueError(f"x_t must have shape [batch, {self.action_dim}], got {tuple(x_t.shape)}")
        x_one_hot = F.one_hot(x_t.long(), num_classes=2).float()
        static_features = self.static_edge_features.to(x_t.device).unsqueeze(0).expand(x_t.shape[0], -1, -1)
        edge_time = time_embedding.unsqueeze(1).expand(-1, self.action_dim, -1)
        return torch.cat([x_one_hot, static_features, edge_time], dim=-1)

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(time)
        _node_hidden, edge_repr, _graph_repr = self.encoder(
            self._node_features(state, time_embedding),
            self._edge_features(x_t, time_embedding),
            self.edge_index,
        )
        return self.edge_head(edge_repr)


class GNNCategoricalDiffusion(nn.Module):
    """GNN actor：categorical diffusion + 每步径向投影。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_timesteps: int,
        branch_data: Sequence[Dict[str, float]],
        num_buses: int,
        max_corruption: float = 0.5,
        sample_temperature: float = 1.0,
        hidden_size: int = 128,
        num_layers: int = 2,
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
        self.model = GNNCategoricalTopologyModel(
            state_dim=state_dim,
            action_dim=action_dim,
            branch_data=branch_data,
            num_buses=num_buses,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

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
