from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class TopologyValidationResult:
    is_feasible: bool
    reasons: List[str]
    closed_switches: int


class _UnionFind:
    """用于径向拓扑校验，显式检查连通性与环路约束。"""

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


def validate_radial_topology(
    topology: Sequence[float],
    branch_data: Sequence[Dict[str, float]],
    num_buses: int,
) -> TopologyValidationResult:
    """校验 0/1 拓扑是否为配电网可执行的径向连通网络。

    当前阶段只校验图约束，不把潮流越限混入这里；潮流约束应由环境或后续
    reward/feasibility 模块返回，避免把两个不同失败原因混在一起。
    """

    topology_array = np.asarray(topology).reshape(-1)
    reasons: List[str] = []

    if topology_array.shape[0] != len(branch_data):
        reasons.append(
            f"dimension_mismatch_expected_{len(branch_data)}_got_{topology_array.shape[0]}"
        )
        return TopologyValidationResult(False, reasons, int(np.sum(topology_array > 0.5)))

    rounded = np.rint(topology_array).astype(int)
    if np.any((rounded != 0) & (rounded != 1)):
        reasons.append("non_binary_values")

    closed_switches = int(np.sum(rounded))
    expected_edges = num_buses - 1
    if closed_switches != expected_edges:
        reasons.append(f"edge_count_{closed_switches}_expected_{expected_edges}")

    union_find = _UnionFind(num_buses)
    has_cycle = False
    for branch_idx, switch_state in enumerate(rounded):
        if switch_state != 1:
            continue
        node_from = int(branch_data[branch_idx]["from"])
        node_to = int(branch_data[branch_idx]["to"])
        if not union_find.union(node_from, node_to):
            has_cycle = True

    if has_cycle:
        reasons.append("cycle_detected")

    root = union_find.find(0)
    island_nodes = [node for node in range(num_buses) if union_find.find(node) != root]
    if island_nodes:
        preview = ",".join(str(node + 1) for node in island_nodes[:8])
        reasons.append(f"island_nodes_{preview}")

    return TopologyValidationResult(
        is_feasible=len(reasons) == 0,
        reasons=reasons,
        closed_switches=closed_switches,
    )
