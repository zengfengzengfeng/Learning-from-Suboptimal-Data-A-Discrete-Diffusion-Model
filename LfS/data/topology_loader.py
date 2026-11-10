"""Topology action loader for the 33-node distribution reconfiguration task.

The Topology_History sheet stores one topology per time column:
- column 1: start node
- column 2: end node
- column 3..N: switch status for each historical scenario/time step

In the LfS adaptation, topology is treated as the action. The historical
binary switch states are exposed as float32 action vectors, so later diffusion
models can use the same interface for continuous scores in [0, 1].
"""

from __future__ import annotations

import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from xml.etree import ElementTree as ET

import numpy as np

try:
    from data.dataset_loader import TimeIndex
except ModuleNotFoundError:
    # 允许直接执行 python data/topology_loader.py 做阶段性验收；
    # 训练代码从项目根目录导入 data.topology_loader 时不会进入这里。
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data.dataset_loader import TimeIndex


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


@dataclass(frozen=True)
class SwitchBranch:
    """A controllable switch branch in the 33-node test system."""

    index: int
    start_node: int
    end_node: int

    @property
    def edge(self) -> tuple[int, int]:
        return (self.start_node, self.end_node)


@dataclass(frozen=True)
class TopologyActionSample:
    """Single historical topology action aligned to one time step."""

    time: TimeIndex
    action_raw: np.ndarray

    @property
    def topology(self) -> np.ndarray:
        """Alias used by downstream topology modules."""

        return self.action_raw


@dataclass
class TopologyActionDataset:
    """Training-friendly topology action dataset.

    action_matrix shape is [num_steps, num_switches]. Each row is one topology
    action and each column corresponds to the switch branch with the same index
    in branches.
    """

    branches: List[SwitchBranch]
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
        return len(self.time_index)

    def sample_at(self, step_idx: int) -> TopologyActionSample:
        if step_idx < 0 or step_idx >= self.num_steps:
            raise IndexError(
                f"step_idx out of range: {step_idx}, valid range is [0, {self.num_steps - 1}]"
            )
        return TopologyActionSample(
            time=self.time_index[step_idx],
            action_raw=self.action_matrix[step_idx].copy(),
        )

    def action_at(self, step_idx: int) -> np.ndarray:
        return self.sample_at(step_idx).action_raw

    def branch_status_at(self, step_idx: int) -> Dict[tuple[int, int], float]:
        action = self.action_at(step_idx)
        return {branch.edge: float(action[branch.index]) for branch in self.branches}

    def to_binary_action(self, step_idx: int, threshold: float = 0.5) -> np.ndarray:
        """Decode continuous switch scores to binary open/closed states.

        This is only a deterministic helper for inspection or later hard
        topology execution. The loader itself keeps the original float action
        representation for diffusion training.
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        return (self.action_at(step_idx) >= threshold).astype(np.float32)


class TopologyXlsxReader:
    """Small xlsx reader based on Python standard library.

    Pandas/openpyxl can read this file too, but the existing dataset loader uses
    direct xlsx XML parsing. Keeping the same style makes this module easy to
    call in lightweight training environments.
    """

    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.file_path}")

    def read_sheet_rows(self, sheet_name: str) -> List[List[str]]:
        with zipfile.ZipFile(self.file_path) as zf:
            shared_strings = self._read_shared_strings(zf)
            sheet_path = self._resolve_sheet_path(zf, sheet_name)
            root = ET.fromstring(zf.read(sheet_path))
            sheet_data = root.find(f"{{{NS_MAIN}}}sheetData")
            if sheet_data is None:
                raise ValueError(f"Sheet {sheet_name} has no sheetData: {self.file_path}")

            rows: List[List[str]] = []
            for row in sheet_data.findall(f"{{{NS_MAIN}}}row"):
                cell_map: Dict[int, str] = {}
                max_col_idx = -1
                for cell in row.findall(f"{{{NS_MAIN}}}c"):
                    ref = cell.attrib.get("r")
                    if ref is None:
                        raise ValueError(f"Sheet {sheet_name} contains a cell without coordinate")

                    col_idx = self._column_letters_to_index(self._extract_column_letters(ref))
                    cell_map[col_idx] = self._read_cell_value(cell, shared_strings)
                    max_col_idx = max(max_col_idx, col_idx)

                dense_row = ["" for _ in range(max_col_idx + 1)]
                for col_idx, value in cell_map.items():
                    dense_row[col_idx] = value
                rows.append(dense_row)

            return rows

    def _read_shared_strings(self, zf: zipfile.ZipFile) -> List[str]:
        if "xl/sharedStrings.xml" not in zf.namelist():
            return []

        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        shared_strings: List[str] = []
        for si in root.findall(f"{{{NS_MAIN}}}si"):
            text_parts = [node.text or "" for node in si.iterfind(f".//{{{NS_MAIN}}}t")]
            shared_strings.append("".join(text_parts))
        return shared_strings

    def _resolve_sheet_path(self, zf: zipfile.ZipFile, sheet_name: str) -> str:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = workbook.find(f"{{{NS_MAIN}}}sheets")
        if sheets is None:
            raise ValueError(f"Workbook has no sheets definition: {self.file_path}")

        relationship_id: Optional[str] = None
        for sheet in sheets:
            if sheet.attrib.get("name") == sheet_name:
                relationship_id = sheet.attrib.get(f"{{{NS_REL}}}id")
                break

        if relationship_id is None:
            raise ValueError(f"Sheet {sheet_name} not found in {self.file_path.name}")

        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        for rel in rels:
            if rel.attrib.get("Id") == relationship_id:
                target = rel.attrib["Target"].lstrip("/")
                return target if target.startswith("xl/") else f"xl/{target}"

        raise ValueError(f"Cannot resolve relationship id {relationship_id} for {sheet_name}")

    @staticmethod
    def _extract_column_letters(cell_ref: str) -> str:
        match = re.match(r"([A-Z]+)", cell_ref)
        if match is None:
            raise ValueError(f"Invalid cell reference: {cell_ref}")
        return match.group(1)

    @staticmethod
    def _column_letters_to_index(column_letters: str) -> int:
        index = 0
        for char in column_letters:
            index = index * 26 + (ord(char) - ord("A") + 1)
        return index - 1

    @staticmethod
    def _read_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> str:
        if cell.attrib.get("t") == "inlineStr":
            text_node = cell.find(f".//{{{NS_MAIN}}}t")
            return "" if text_node is None else (text_node.text or "")

        value_node = cell.find(f"{{{NS_MAIN}}}v")
        if value_node is None:
            return ""

        raw_value = value_node.text or ""
        if cell.attrib.get("t") == "s":
            return shared_strings[int(raw_value)]
        return raw_value


def load_topology_action_dataset(
    base_dir: Path | str | None = None,
    topology_file_name: str = "Optimization_Detailed_Report_33.xlsx",
    sheet_name: str = "Topology_History",
    expected_switches: int = 37,
    expected_steps: int = 4800,
    validate_binary_history: bool = True,
) -> TopologyActionDataset:
    """Load historical topology actions for LfS training.

    Returned action_matrix has shape [4800, 37] for the provided 33-node file.
    Values are float32 in [0, 1]. Historical records are expected to be binary;
    validate_binary_history can be disabled only when reading model-generated
    continuous score files with the same table format.
    """

    if base_dir is None:
        base_dir = get_default_project_root()

    file_path = Path(base_dir) / topology_file_name
    rows = TopologyXlsxReader(file_path).read_sheet_rows(sheet_name)
    data_rows = _select_topology_data_rows(rows, sheet_name)

    branches = _parse_switch_branches(data_rows, sheet_name)
    action_matrix = _parse_action_matrix(data_rows, sheet_name)

    _validate_shape(
        action_matrix=action_matrix,
        branches=branches,
        expected_switches=expected_switches,
        expected_steps=expected_steps,
        sheet_name=sheet_name,
    )
    _validate_action_values(
        action_matrix=action_matrix,
        validate_binary_history=validate_binary_history,
        sheet_name=sheet_name,
    )

    time_index = [_build_time_index(global_step) for global_step in range(1, action_matrix.shape[0] + 1)]
    return TopologyActionDataset(
        branches=branches,
        time_index=time_index,
        action_matrix=action_matrix,
    )


def get_topology_action_sample(
    step_idx: int,
    base_dir: Path | str | None = None,
) -> TopologyActionSample:
    dataset = load_topology_action_dataset(base_dir=base_dir)
    return dataset.sample_at(step_idx)


def get_topology_action(
    step_idx: int,
    base_dir: Path | str | None = None,
) -> np.ndarray:
    """主训练流程的便捷接口：直接返回某个时间步的动作向量。"""

    return get_topology_action_sample(step_idx=step_idx, base_dir=base_dir).action_raw


def load_topology_action_matrix(
    base_dir: Path | str | None = None,
) -> np.ndarray:
    """主训练流程的便捷接口：直接返回 [num_steps, action_dim] 动作矩阵。"""

    dataset = load_topology_action_dataset(base_dir=base_dir)
    return dataset.action_matrix.copy()


def get_default_project_root() -> Path:
    """返回项目根目录，默认 Excel 数据文件放在这里。"""

    return Path(__file__).resolve().parents[1]


def _select_topology_data_rows(rows: Sequence[Sequence[str]], sheet_name: str) -> List[List[str]]:
    data_rows: List[List[str]] = []
    for row_idx, row in enumerate(rows, start=1):
        if len(row) < 3 or _is_empty_row(row):
            continue

        first = str(row[0]).strip().lower()
        second = str(row[1]).strip().lower()
        if first in {"from_node", "from", "start_node"} and second in {"to_node", "to", "end_node"}:
            continue

        if _can_parse_int(row[0]) and _can_parse_int(row[1]):
            data_rows.append(list(row))
            continue

        raise ValueError(
            f"{sheet_name} row {row_idx} is not a valid topology row: "
            f"first two columns must be start/end node, got {row[:2]}"
        )

    if not data_rows:
        raise ValueError(f"{sheet_name} contains no valid topology data rows")
    return data_rows


def _parse_switch_branches(data_rows: Sequence[Sequence[str]], sheet_name: str) -> List[SwitchBranch]:
    branches: List[SwitchBranch] = []
    seen_edges: set[tuple[int, int]] = set()

    for switch_idx, row in enumerate(data_rows):
        start_node = _to_int(row[0], f"{sheet_name} switch {switch_idx} start_node")
        end_node = _to_int(row[1], f"{sheet_name} switch {switch_idx} end_node")
        edge = (start_node, end_node)
        if edge in seen_edges:
            raise ValueError(f"{sheet_name} contains duplicated switch branch: {edge}")
        seen_edges.add(edge)
        branches.append(SwitchBranch(index=switch_idx, start_node=start_node, end_node=end_node))

    return branches


def _parse_action_matrix(data_rows: Sequence[Sequence[str]], sheet_name: str) -> np.ndarray:
    expected_steps = len(data_rows[0]) - 2
    if expected_steps <= 0:
        raise ValueError(f"{sheet_name} has no topology status columns")

    switch_rows: List[List[float]] = []
    for switch_idx, row in enumerate(data_rows):
        raw_values = list(row[2:])
        if len(raw_values) != expected_steps:
            raise ValueError(
                f"{sheet_name} switch {switch_idx} has inconsistent status columns: "
                f"expected {expected_steps}, got {len(raw_values)}"
            )

        switch_rows.append(
            [
                _to_float(value, f"{sheet_name} switch {switch_idx} time column {time_col + 1}")
                for time_col, value in enumerate(raw_values)
            ]
        )

    # Excel layout is [switch, time]. Training uses [time, action_dim].
    return np.asarray(switch_rows, dtype=np.float32).T


def _validate_shape(
    action_matrix: np.ndarray,
    branches: Sequence[SwitchBranch],
    expected_switches: int,
    expected_steps: int,
    sheet_name: str,
) -> None:
    if len(branches) != expected_switches:
        raise ValueError(
            f"{sheet_name} switch count mismatch: expected {expected_switches}, got {len(branches)}"
        )

    if action_matrix.shape != (expected_steps, expected_switches):
        raise ValueError(
            f"{sheet_name} action_matrix shape mismatch: expected "
            f"({expected_steps}, {expected_switches}), got {action_matrix.shape}"
        )


def _validate_action_values(
    action_matrix: np.ndarray,
    validate_binary_history: bool,
    sheet_name: str,
) -> None:
    if np.isnan(action_matrix).any():
        raise ValueError(f"{sheet_name} contains NaN switch status values")

    if (action_matrix < 0.0).any() or (action_matrix > 1.0).any():
        min_value = float(action_matrix.min())
        max_value = float(action_matrix.max())
        raise ValueError(
            f"{sheet_name} switch status must be in [0, 1], got min={min_value}, max={max_value}"
        )

    if validate_binary_history and not np.isin(action_matrix, [0.0, 1.0]).all():
        bad_positions = np.argwhere(~np.isin(action_matrix, [0.0, 1.0]))
        first_step, first_switch = bad_positions[0]
        bad_value = float(action_matrix[first_step, first_switch])
        raise ValueError(
            f"{sheet_name} historical topology must be binary, first non-binary value "
            f"is {bad_value} at step={first_step}, switch={first_switch}"
        )


def _build_time_index(global_step: int) -> TimeIndex:
    if global_step <= 0:
        raise ValueError(f"global_step must be positive, got {global_step}")

    return TimeIndex(
        label=f"H{global_step}",
        global_step=global_step,
        day_index=(global_step - 1) // 24 + 1,
        hour_of_day=(global_step - 1) % 24 + 1,
    )


def _is_empty_row(row: Sequence[str]) -> bool:
    return all(str(value).strip() == "" for value in row)


def _can_parse_int(value: str) -> bool:
    try:
        int(float(str(value).strip()))
    except ValueError:
        return False
    return True


def _to_int(value: str, field_name: str) -> int:
    value = str(value).strip()
    if value == "":
        raise ValueError(f"{field_name} is empty")
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} cannot be converted to int: {value}") from exc


def _to_float(value: str, field_name: str) -> float:
    value = str(value).strip()
    if value == "":
        raise ValueError(f"{field_name} is empty")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} cannot be converted to float: {value}") from exc


def _print_dataset_preview(dataset: TopologyActionDataset, preview_count: int = 3) -> None:
    preview_count = min(preview_count, dataset.num_steps)
    print("Topology action dataset loaded")
    print(f"num_steps: {dataset.num_steps}")
    print(f"num_switches/action_dim: {dataset.action_dim}")
    print(f"action_matrix shape: {dataset.action_matrix.shape}")
    print(f"first 5 branches: {[branch.edge for branch in dataset.branches[:5]]}")

    for step_idx in range(preview_count):
        sample = dataset.sample_at(step_idx)
        print(
            f"[step {step_idx}] time={sample.time.label}, "
            f"day={sample.time.day_index}, hour={sample.time.hour_of_day}, "
            f"action[:10]={sample.action_raw[:10].tolist()}"
        )


# Main-program call examples:
# from data.topology_loader import (
#     get_topology_action,
#     get_topology_action_sample,
#     load_topology_action_dataset,
#     load_topology_action_matrix,
# )
#
# topology_dataset = load_topology_action_dataset()
# action_dim = topology_dataset.action_dim
# first_action = topology_dataset.action_at(0)
# action_matrix = load_topology_action_matrix()
# first_sample = get_topology_action_sample(step_idx=0)
# first_action_direct = get_topology_action(step_idx=0)
#
if __name__ == "__main__":
    dataset = load_topology_action_dataset()
    _print_dataset_preview(dataset)
