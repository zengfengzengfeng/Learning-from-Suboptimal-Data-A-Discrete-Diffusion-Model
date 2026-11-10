from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from env_setting.MISOCP_192 import GRID_NODES


@dataclass(frozen=True)
class TopologyRepairResult:
    topology: np.ndarray
    selected_branch_indices: np.ndarray
    repair_mode: str


class _UnionFind:
    """Kruskal 修复需要的并查集，并记录每个分量包含的电源节点数量。"""

    def __init__(self, n: int, grid_nodes: Sequence[int]):
        self.parent = list(range(n))
        self.rank = [0] * n
        grid_node_set = {int(node) for node in grid_nodes}
        self.grid_count = [1 if node in grid_node_set else 0 for node in range(n)]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def can_union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        return self.grid_count[root_a] + self.grid_count[root_b] <= 1

    def union(self, a: int, b: int) -> bool:
        if not self.can_union(a, b):
            return False

        root_a = self.find(a)
        root_b = self.find(b)
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
            self.grid_count[root_b] += self.grid_count[root_a]
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
            self.grid_count[root_a] += self.grid_count[root_b]
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1
            self.grid_count[root_a] += self.grid_count[root_b]
        return True


def repair_scores_to_radial_topology(
    action_scores: Sequence[float],
    branch_data: Sequence[Dict[str, float]],
    num_buses: int,
    grid_nodes: Sequence[int] = GRID_NODES,
) -> TopologyRepairResult:
    """把连续支路分数修复成 192 节点双电源径向森林。

    actor/diffusion 输出的是每条支路的优先级。这里按分数降序选择支路，
    但禁止两个已经含有电源节点的分量相连；最终闭合边数为
    NB - len(GRID_NODES)，从图结构上避免 175、189 被并到同一棵树中。
    """

    scores = np.asarray(action_scores, dtype=np.float32).reshape(-1)
    if scores.shape[0] != len(branch_data):
        raise ValueError(
            f"action_scores dimension must be {len(branch_data)}, got {scores.shape[0]}"
        )

    grid_node_set = {int(node) for node in grid_nodes}
    invalid_grid_nodes = [node for node in grid_node_set if node < 0 or node >= num_buses]
    if invalid_grid_nodes:
        raise ValueError(f"grid_nodes out of range for num_buses={num_buses}: {invalid_grid_nodes}")

    expected_edges = num_buses - len(grid_node_set)
    ranking = np.argsort(-scores, kind="mergesort")
    union_find = _UnionFind(num_buses, tuple(grid_node_set))
    topology = np.zeros(len(branch_data), dtype=np.float32)
    selected = []

    for branch_idx in ranking:
        node_from = int(branch_data[branch_idx]["from"])
        node_to = int(branch_data[branch_idx]["to"])
        if not 0 <= node_from < num_buses or not 0 <= node_to < num_buses:
            raise ValueError(
                f"branch {branch_idx + 1} endpoint out of range: "
                f"{node_from + 1}->{node_to + 1}, num_buses={num_buses}"
            )
        if union_find.union(node_from, node_to):
            topology[branch_idx] = 1.0
            selected.append(int(branch_idx))
            if len(selected) == expected_edges:
                break

    if len(selected) != expected_edges:
        raise RuntimeError(
            "failed to repair action into two-root radial forest: "
            f"selected_edges={len(selected)}, expected={expected_edges}, "
            f"grid_nodes={[node + 1 for node in sorted(grid_node_set)]}"
        )

    return TopologyRepairResult(
        topology=topology,
        selected_branch_indices=np.asarray(selected, dtype=np.int64),
        repair_mode="score_kruskal_two_root_forest",
    )
