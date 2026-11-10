from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class TopologyRepairResult:
    topology: np.ndarray
    selected_branch_indices: np.ndarray
    repair_mode: str


class _UnionFind:
    """Kruskal 生成树修复需要的并查集。"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1
        return True


def repair_scores_to_radial_topology(
    action_scores: Sequence[float],
    branch_data: Sequence[Dict[str, float]],
    num_buses: int,
) -> TopologyRepairResult:
    """把连续支路优先级分数修复成一棵径向生成树。

    diffusion 更适合输出连续潜在动作，本函数按分数降序选择不会成环的支路，
    保证输出满足配电网重构的最基础硬约束：连通、无环、无孤岛。
    """

    scores = np.asarray(action_scores, dtype=np.float32).reshape(-1)
    if scores.shape[0] != len(branch_data):
        raise ValueError(
            f"action_scores dimension must be {len(branch_data)}, got {scores.shape[0]}"
        )

    ranking = np.argsort(-scores, kind="mergesort")
    union_find = _UnionFind(num_buses)
    topology = np.zeros(len(branch_data), dtype=np.float32)
    selected = []

    for branch_idx in ranking:
        node_from = int(branch_data[branch_idx]["from"])
        node_to = int(branch_data[branch_idx]["to"])
        if union_find.union(node_from, node_to):
            topology[branch_idx] = 1.0
            selected.append(int(branch_idx))
            if len(selected) == num_buses - 1:
                break

    if len(selected) != num_buses - 1:
        raise RuntimeError(
            "failed to repair action into radial topology: "
            f"selected_edges={len(selected)}, expected={num_buses - 1}"
        )

    return TopologyRepairResult(
        topology=topology,
        selected_branch_indices=np.asarray(selected, dtype=np.int64),
        repair_mode="score_kruskal",
    )
