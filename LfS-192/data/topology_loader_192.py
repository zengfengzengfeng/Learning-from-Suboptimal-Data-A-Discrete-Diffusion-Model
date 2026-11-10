"""Topology action loader for the 192-bus reconfiguration task.

The Topology_History sheet stores one topology per scenario column:
- column 1: From_Node
- column 2: To_Node
- column 3..N: binary switch status for each scenario

Unlike the 33-bus case, the 192-bus system has two grid nodes. A valid topology
is therefore a two-root radial forest, not a single tree:
- closed branch count = 192 - 2 = 190
- every connected component must contain exactly one grid node
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


NB = 192
GRID_BUS_IDS = (175, 189)
GRID_NODES = tuple(bus_id - 1 for bus_id in GRID_BUS_IDS)
EXPECTED_CLOSED_BRANCHES = NB - len(GRID_NODES)


@dataclass(frozen=True)
class TimeIndex:
    label: str
    global_step: int
    day_index: int
    hour_of_day: int


@dataclass(frozen=True)
class SwitchBranch192:
    index: int
    start_node: int
    end_node: int

    @property
    def edge(self) -> tuple[int, int]:
        return (self.start_node, self.end_node)


@dataclass(frozen=True)
class TopologyActionSample192:
    time: TimeIndex
    action_raw: np.ndarray

    @property
    def topology(self) -> np.ndarray:
        return self.action_raw


@dataclass
class TopologyActionDataset192:
    branches: List[SwitchBranch192]
    time_index: List[TimeIndex]
    action_matrix: np.ndarray

    @property
    def num_switches(self) -> int:
        return len(self.branches)

    @property
    def action_dim(self) -> int:
        return self.num_switches

    @property
    def num_steps(self) -> int:
        return self.action_matrix.shape[0]

    def sample_at(self, step_idx: int) -> TopologyActionSample192:
        if not 0 <= step_idx < self.num_steps:
            raise IndexError(f"step_idx out of range: {step_idx}, valid range is [0, {self.num_steps - 1}]")
        return TopologyActionSample192(
            time=self.time_index[step_idx],
            action_raw=self.action_matrix[step_idx].copy(),
        )

    def action_at(self, step_idx: int) -> np.ndarray:
        return self.sample_at(step_idx).action_raw

    def branch_status_at(self, step_idx: int) -> Dict[tuple[int, int], float]:
        action = self.action_at(step_idx)
        return {branch.edge: float(action[branch.index]) for branch in self.branches}


class _UnionFind:
    def __init__(self, node_count: int):
        self.parent = list(range(node_count))
        self.rank = [0] * node_count

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, node_a: int, node_b: int) -> bool:
        root_a = self.find(node_a)
        root_b = self.find(node_b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return True


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_topology_action_dataset_192(
    base_dir: Path | str | None = None,
    topology_file_name: str = "Optimization_Detailed_Report_192.xlsx",
    sheet_name: str = "Topology_History",
    expected_switches: int | None = None,
    expected_steps: int | None = None,
    validate_binary_history: bool = True,
    validate_radial_forest: bool = True,
) -> TopologyActionDataset192:
    """Load 192-bus topology actions from a Topology_History sheet."""

    if base_dir is None:
        base_dir = get_default_project_root()
    file_path = Path(base_dir) / topology_file_name
    if not file_path.exists():
        raise FileNotFoundError(f"topology file not found: {file_path}")

    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    if df.shape[1] < 3:
        raise ValueError(f"{file_path} {sheet_name} must contain From_Node, To_Node and at least one topology column")

    first_columns = [str(column).strip().lower() for column in df.columns[:2]]
    if first_columns[0] not in {"from_node", "from", "start_node"}:
        raise ValueError(f"{file_path} {sheet_name} first column must be From_Node/from/start_node")
    if first_columns[1] not in {"to_node", "to", "end_node"}:
        raise ValueError(f"{file_path} {sheet_name} second column must be To_Node/to/end_node")

    branches = _parse_switch_branches(df, file_path, sheet_name)
    action_matrix = df.iloc[:, 2:].to_numpy(dtype=np.float64).T.astype(np.float32)
    labels = [str(column) for column in df.columns[2:]]

    if expected_switches is not None and len(branches) != expected_switches:
        raise ValueError(f"{file_path} {sheet_name} switch count mismatch: expected {expected_switches}, got {len(branches)}")
    if expected_steps is not None and action_matrix.shape[0] != expected_steps:
        raise ValueError(f"{file_path} {sheet_name} step count mismatch: expected {expected_steps}, got {action_matrix.shape[0]}")
    if not np.isfinite(action_matrix).all():
        raise ValueError(f"{file_path} {sheet_name} contains NaN or infinite topology values")
    if validate_binary_history and not np.isin(action_matrix, [0.0, 1.0]).all():
        unique_values = sorted(np.unique(action_matrix).astype(float).tolist())
        raise ValueError(f"{file_path} {sheet_name} contains non-binary topology values: {unique_values[:20]}")
    if validate_radial_forest:
        _validate_all_topologies(action_matrix, branches, file_path, sheet_name)

    time_index = [_build_time_index(label, idx) for idx, label in enumerate(labels)]
    return TopologyActionDataset192(
        branches=branches,
        time_index=time_index,
        action_matrix=action_matrix,
    )


def get_topology_action_sample_192(
    step_idx: int,
    base_dir: Path | str | None = None,
    topology_file_name: str = "Optimization_Detailed_Report_192.xlsx",
) -> TopologyActionSample192:
    dataset = load_topology_action_dataset_192(base_dir=base_dir, topology_file_name=topology_file_name)
    return dataset.sample_at(step_idx)


def get_topology_action_192(
    step_idx: int,
    base_dir: Path | str | None = None,
    topology_file_name: str = "Optimization_Detailed_Report_192.xlsx",
) -> np.ndarray:
    return get_topology_action_sample_192(step_idx, base_dir, topology_file_name).action_raw


def load_topology_action_matrix_192(
    base_dir: Path | str | None = None,
    topology_file_name: str = "Optimization_Detailed_Report_192.xlsx",
) -> np.ndarray:
    return load_topology_action_dataset_192(base_dir=base_dir, topology_file_name=topology_file_name).action_matrix.copy()


def is_valid_two_root_radial_forest(topology: np.ndarray, branches: List[SwitchBranch192]) -> bool:
    topology = np.asarray(topology, dtype=int).reshape(-1)
    if topology.shape[0] != len(branches):
        return False
    if not set(np.unique(topology).tolist()).issubset({0, 1}):
        return False
    if int(np.sum(topology)) != EXPECTED_CLOSED_BRANCHES:
        return False

    union_find = _UnionFind(NB)
    for branch_index, closed in enumerate(topology):
        if closed == 0:
            continue
        branch = branches[branch_index]
        if not union_find.union(branch.start_node - 1, branch.end_node - 1):
            return False

    component_grid_counts: dict[int, int] = {}
    for node in range(NB):
        root = union_find.find(node)
        component_grid_counts.setdefault(root, 0)
    for grid_node in GRID_NODES:
        root = union_find.find(grid_node)
        component_grid_counts[root] += 1
    return all(grid_count == 1 for grid_count in component_grid_counts.values())


def _parse_switch_branches(df: pd.DataFrame, file_path: Path, sheet_name: str) -> List[SwitchBranch192]:
    branches: List[SwitchBranch192] = []
    seen_edges: set[tuple[int, int]] = set()
    for index, row in df.iterrows():
        start_node = int(row.iloc[0])
        end_node = int(row.iloc[1])
        if not 1 <= start_node <= NB or not 1 <= end_node <= NB:
            raise ValueError(f"{file_path} {sheet_name} row {index + 2} has out-of-range branch {start_node}->{end_node}")
        edge = (start_node, end_node)
        if edge in seen_edges:
            raise ValueError(f"{file_path} {sheet_name} contains duplicated branch: {edge}")
        seen_edges.add(edge)
        branches.append(SwitchBranch192(index=len(branches), start_node=start_node, end_node=end_node))
    return branches


def _validate_all_topologies(
    action_matrix: np.ndarray,
    branches: List[SwitchBranch192],
    file_path: Path,
    sheet_name: str,
) -> None:
    invalid_steps: List[str] = []
    for step_idx, topology in enumerate(action_matrix):
        if not is_valid_two_root_radial_forest(topology, branches):
            invalid_steps.append(f"{step_idx}:{int(np.sum(topology))}")
            if len(invalid_steps) >= 10:
                break
    if invalid_steps:
        raise ValueError(
            f"{file_path} {sheet_name} contains invalid two-root radial forest topologies "
            f"(step:closed_count): {invalid_steps}"
        )


def _build_time_index(label: str, fallback_index: int) -> TimeIndex:
    label = str(label)
    if label.upper().startswith("H"):
        try:
            global_step = int(label[1:])
        except ValueError:
            global_step = fallback_index
    else:
        global_step = fallback_index
    return TimeIndex(
        label=label,
        global_step=global_step,
        day_index=global_step // 24 + 1,
        hour_of_day=global_step % 24 + 1,
    )


if __name__ == "__main__":
    dataset = load_topology_action_dataset_192()
    sample = dataset.sample_at(0)
    print("192-bus topology dataset loaded")
    print(f"steps={dataset.num_steps}, switches={dataset.num_switches}")
    print(f"first={sample.time.label}, closed={int(np.sum(sample.action_raw))}")
