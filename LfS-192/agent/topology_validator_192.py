from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from env_setting.MISOCP_192 import GRID_NODES


@dataclass(frozen=True)
class TopologyValidationResult:
    is_feasible: bool
    reasons: List[str]
    closed_switches: int


class _UnionFind:
    """用于拓扑校验的并查集，显式检查成环和连通分量。"""

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
    grid_nodes: Sequence[int] = GRID_NODES,
) -> TopologyValidationResult:
    """校验 192 节点双电源径向森林。

    192 系统中 175、189 为上级电网节点，因此合法拓扑不是单棵树，
    而是每个连通分量恰好包含一个电源节点的径向森林。这里仅检查
    图约束；潮流、电压、电流越限仍交给 Env192/SOCP192 评估。
    """

    topology_array = np.asarray(topology).reshape(-1)
    reasons: List[str] = []
    grid_node_set = {int(node) for node in grid_nodes}

    if topology_array.shape[0] != len(branch_data):
        reasons.append(
            f"dimension_mismatch_expected_{len(branch_data)}_got_{topology_array.shape[0]}"
        )
        return TopologyValidationResult(False, reasons, int(np.sum(topology_array > 0.5)))
    if not grid_node_set:
        reasons.append("empty_grid_nodes")
        return TopologyValidationResult(False, reasons, int(np.sum(topology_array > 0.5)))

    invalid_grid_nodes = [node for node in grid_node_set if node < 0 or node >= num_buses]
    if invalid_grid_nodes:
        reasons.append(f"grid_nodes_out_of_range_{invalid_grid_nodes}")

    rounded = np.rint(topology_array).astype(int)
    if np.any((rounded != 0) & (rounded != 1)):
        reasons.append("non_binary_values")

    closed_switches = int(np.sum(rounded))
    expected_edges = num_buses - len(grid_node_set)
    if closed_switches != expected_edges:
        reasons.append(f"edge_count_{closed_switches}_expected_{expected_edges}")

    union_find = _UnionFind(num_buses)
    has_cycle = False
    for branch_idx, switch_state in enumerate(rounded):
        if switch_state != 1:
            continue
        node_from = int(branch_data[branch_idx]["from"])
        node_to = int(branch_data[branch_idx]["to"])
        if not 0 <= node_from < num_buses or not 0 <= node_to < num_buses:
            reasons.append(f"branch_{branch_idx + 1}_endpoint_out_of_range")
            continue
        if not union_find.union(node_from, node_to):
            has_cycle = True

    if has_cycle:
        reasons.append("cycle_detected")

    components: Dict[int, List[int]] = {}
    for node in range(num_buses):
        components.setdefault(union_find.find(node), []).append(node)

    for nodes in components.values():
        component_grid_nodes = [node for node in nodes if node in grid_node_set]
        if len(component_grid_nodes) == 0:
            preview = ",".join(str(node + 1) for node in nodes[:8])
            reasons.append(f"component_without_grid_{preview}")
        elif len(component_grid_nodes) > 1:
            preview = ",".join(str(node + 1) for node in component_grid_nodes)
            reasons.append(f"component_with_multiple_grids_{preview}")

    return TopologyValidationResult(
        is_feasible=len(reasons) == 0,
        reasons=reasons,
        closed_switches=closed_switches,
    )
