from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from agent.helpers import init_weights


def build_edge_index(branch_data: Sequence[Dict[str, float]], num_buses: int) -> torch.Tensor:
    edge_index = []
    for branch_idx, branch in enumerate(branch_data):
        node_from = int(branch["from"])
        node_to = int(branch["to"])
        if not 0 <= node_from < num_buses or not 0 <= node_to < num_buses:
            raise ValueError(
                f"branch {branch_idx} endpoint out of range: "
                f"from={node_from}, to={node_to}, num_buses={num_buses}"
            )
        edge_index.append((node_from, node_to))
    return torch.as_tensor(edge_index, dtype=torch.long).t().contiguous()


def build_static_edge_features(branch_data: Sequence[Dict[str, float]]) -> torch.Tensor:
    rows = []
    for branch_idx, branch in enumerate(branch_data):
        if "r" not in branch or "x" not in branch:
            raise ValueError(f"branch {branch_idx} missing r/x fields")
        rows.append([float(branch["r"]), float(branch["x"])])
    return torch.as_tensor(rows, dtype=torch.float32)


class TopologyGraphLayer(nn.Module):
    """基于支路消息传递的轻量 GNN 层。

    这里不依赖 torch_geometric，避免给现有训练环境增加额外安装要求。每条支路按无向边处理，
    同一组边特征同时向 from/to 两端聚合，使 critic/actor 能感知配电网连通关系。
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        message_dim = hidden_dim * 2 + edge_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_hidden: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if node_hidden.dim() != 3:
            raise ValueError(f"node_hidden must have shape [batch, buses, hidden], got {tuple(node_hidden.shape)}")
        if edge_attr.dim() != 3:
            raise ValueError(f"edge_attr must have shape [batch, branches, edge_dim], got {tuple(edge_attr.shape)}")

        src = edge_index[0]
        dst = edge_index[1]
        src_hidden = node_hidden[:, src, :]
        dst_hidden = node_hidden[:, dst, :]

        forward_message = self.message_mlp(torch.cat([src_hidden, dst_hidden, edge_attr], dim=-1))
        backward_message = self.message_mlp(torch.cat([dst_hidden, src_hidden, edge_attr], dim=-1))

        aggregated = torch.zeros_like(node_hidden)
        aggregated.index_add_(1, dst, forward_message)
        aggregated.index_add_(1, src, backward_message)

        return self.node_mlp(torch.cat([node_hidden, aggregated], dim=-1))


class TopologyGNNEncoder(nn.Module):
    """把节点状态和支路特征编码为节点/支路/整图表示。"""

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            TopologyGraphLayer(hidden_dim=hidden_dim, edge_dim=hidden_dim) for _ in range(num_layers)
        )
        self.edge_readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(init_weights)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_hidden = self.node_encoder(node_features)
        edge_hidden = self.edge_encoder(edge_features)
        for layer in self.layers:
            node_hidden = node_hidden + layer(node_hidden, edge_hidden, edge_index)

        src = edge_index[0]
        dst = edge_index[1]
        edge_repr = self.edge_readout(torch.cat([node_hidden[:, src, :], node_hidden[:, dst, :], edge_hidden], dim=-1))
        graph_repr = torch.cat([node_hidden.mean(dim=1), edge_repr.mean(dim=1)], dim=-1)
        return node_hidden, edge_repr, graph_repr


class _SingleGNNQ(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        branch_data: Sequence[Dict[str, float]],
        num_buses: int,
        hidden_dim: int,
        num_layers: int,
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
        self.register_buffer("edge_index", build_edge_index(branch_data, num_buses))
        self.register_buffer("static_edge_features", build_static_edge_features(branch_data))
        self.encoder = TopologyGNNEncoder(
            node_input_dim=self.node_feature_dim,
            edge_input_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_weights)

    def _node_features(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape [batch, {self.state_dim}], got {tuple(state.shape)}")
        # state 当前按 [特征块, 节点] 展平，这里还原为每个节点的物理特征。
        return state.view(state.shape[0], self.node_feature_dim, self.num_buses).transpose(1, 2).contiguous()

    def _edge_features(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() != 2 or action.shape[1] != self.action_dim:
            raise ValueError(f"action must have shape [batch, {self.action_dim}], got {tuple(action.shape)}")
        closed_score = ((action + 1.0) * 0.5).clamp(0.0, 1.0).unsqueeze(-1)
        static_features = self.static_edge_features.to(action.device).unsqueeze(0).expand(action.shape[0], -1, -1)
        return torch.cat([closed_score, static_features], dim=-1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _node_hidden, _edge_repr, graph_repr = self.encoder(
            self._node_features(state),
            self._edge_features(action),
            self.edge_index,
        )
        return self.q_head(graph_repr)


class GNNTopologyCritic(nn.Module):
    """双 Q 的拓扑 GNN critic，接口保持与现有 MLP Critic 一致。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        branch_data: Sequence[Dict[str, float]],
        num_buses: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.q1_model = _SingleGNNQ(state_dim, action_dim, branch_data, num_buses, hidden_dim, num_layers)
        self.q2_model = _SingleGNNQ(state_dim, action_dim, branch_data, num_buses, hidden_dim, num_layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1_model(state, action), self.q2_model(state, action)

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1_model(state, action)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
