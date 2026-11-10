"""Dataset loader for the 192-bus distribution reconfiguration task.

The 192-bus workflow stores generated scenarios in:
- data/suboptimal_192/typical load scenario_192.xlsx
  - Sheet3: active load, first column is bus id, remaining columns are scenarios
  - Sheet4: reactive load, same layout
- data/suboptimal_192/typical PV scenario_192.xlsx
  - Sheet3: PV active power limit and grid-node active power limit

This module exposes a training-friendly dataset interface. Public samples use
MW/MVar units, matching env_setting.env_192, while the Excel files are read as
kW/kVar and converted exactly once here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


NB = 192
GRID_BUS_IDS = (175, 189)
DEFAULT_DATA_DIR = Path("data") / "suboptimal_192"


@dataclass(frozen=True)
class TimeIndex:
    """Time/scenario index parsed from labels such as H24."""

    label: str
    global_step: int
    day_index: int
    hour_of_day: int


@dataclass
class ScenarioTable:
    """One bus-by-time scenario table in physical MW/MVar units."""

    name: str
    bus_ids: List[int]
    time_index: List[TimeIndex]
    values_by_bus: Dict[int, List[float]]

    @property
    def num_buses(self) -> int:
        return len(self.bus_ids)

    @property
    def num_steps(self) -> int:
        return len(self.time_index)

    def value_at(self, bus_id: int, step_idx: int) -> float:
        return self.values_by_bus[bus_id][step_idx]

    def to_matrix(self) -> np.ndarray:
        return np.asarray([self.values_by_bus[bus_id] for bus_id in self.bus_ids], dtype=np.float64)


@dataclass(frozen=True)
class StateSample192:
    """Single 192-bus scenario sample in MW/MVar units."""

    time: TimeIndex
    p_load_by_bus: Dict[int, float]
    q_load_by_bus: Dict[int, float]
    pv_by_bus: Dict[int, float]
    grid_p_max_by_bus: Dict[int, float]


@dataclass
class DistributionDataset192:
    """Training-friendly state dataset for the 192-bus system."""

    tables: Dict[str, ScenarioTable]
    samples: List[StateSample192]

    @property
    def p_load(self) -> ScenarioTable:
        return self.tables["p_load"]

    @property
    def q_load(self) -> ScenarioTable:
        return self.tables["q_load"]

    @property
    def pv_p_max(self) -> ScenarioTable:
        return self.tables["pv_p_max"]

    @property
    def grid_p_max(self) -> ScenarioTable:
        return self.tables["grid_p_max"]

    @property
    def num_steps(self) -> int:
        return len(self.samples)

    def sample_at(self, step_idx: int) -> StateSample192:
        if not 0 <= step_idx < self.num_steps:
            raise IndexError(f"step_idx out of range: {step_idx}, valid range is [0, {self.num_steps - 1}]")
        return self.samples[step_idx]


def get_default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_distribution_dataset_192(
    base_dir: Path | str | None = None,
    load_file_name: str = "data/suboptimal_192/typical load scenario_192.xlsx",
    pv_file_name: str = "data/suboptimal_192/typical PV scenario_192.xlsx",
    p_sheet_name: str = "Sheet3",
    q_sheet_name: str = "Sheet4",
    pv_sheet_name: str = "Sheet3",
    expected_buses: int = NB,
    expected_steps: int | None = 4800,
) -> DistributionDataset192:
    """Load 192-bus load/PV scenarios and return aligned state samples."""

    if base_dir is None:
        base_dir = get_default_project_root()
    base_dir = Path(base_dir)

    load_path = base_dir / load_file_name
    pv_path = base_dir / pv_file_name

    p_load = _read_bus_scenario_table(
        path=load_path,
        sheet_name=p_sheet_name,
        table_name="p_load",
        expected_buses=expected_buses,
        expected_steps=expected_steps,
        value_scale=1.0 / 1000.0,
        allow_partial_bus_set=False,
    )
    q_load = _read_bus_scenario_table(
        path=load_path,
        sheet_name=q_sheet_name,
        table_name="q_load",
        expected_buses=expected_buses,
        expected_steps=expected_steps,
        value_scale=1.0 / 1000.0,
        allow_partial_bus_set=False,
    )
    pv_raw = _read_bus_scenario_table(
        path=pv_path,
        sheet_name=pv_sheet_name,
        table_name="pv_raw",
        expected_buses=None,
        expected_steps=expected_steps,
        value_scale=1.0 / 1000.0,
        allow_partial_bus_set=True,
    )

    _validate_time_alignment(p_load, q_load, pv_raw)
    pv_p_max, grid_p_max = _split_pv_and_grid_tables(pv_raw, p_load.time_index)
    tables = {
        "p_load": p_load,
        "q_load": q_load,
        "pv_p_max": pv_p_max,
        "grid_p_max": grid_p_max,
    }
    samples = _build_state_samples(tables)
    return DistributionDataset192(tables=tables, samples=samples)


def get_state_sample_192(step_idx: int, base_dir: Path | str | None = None) -> StateSample192:
    return load_distribution_dataset_192(base_dir=base_dir).sample_at(step_idx)


def _read_bus_scenario_table(
    path: Path,
    sheet_name: str,
    table_name: str,
    expected_buses: int | None,
    expected_steps: int | None,
    value_scale: float,
    allow_partial_bus_set: bool,
) -> ScenarioTable:
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    if df.shape[1] < 2:
        raise ValueError(f"{path} {sheet_name} must contain a bus column and at least one scenario column")

    bus_ids = df.iloc[:, 0].to_numpy(dtype=int)
    if len(set(bus_ids.tolist())) != len(bus_ids):
        raise ValueError(f"{path} {sheet_name} contains duplicated bus ids")

    if expected_buses is not None:
        if df.shape[0] != expected_buses:
            raise ValueError(f"{path} {sheet_name} row count mismatch: expected {expected_buses}, got {df.shape[0]}")
        expected_ids = np.arange(1, expected_buses + 1)
        if not np.array_equal(bus_ids, expected_ids):
            raise ValueError(f"{path} {sheet_name} bus ids must be 1..{expected_buses}")

    if not allow_partial_bus_set and (bus_ids.min() < 1 or bus_ids.max() > NB):
        raise ValueError(f"{path} {sheet_name} contains out-of-range bus ids")
    if allow_partial_bus_set and any(bus_id < 1 or bus_id > NB for bus_id in bus_ids):
        raise ValueError(f"{path} {sheet_name} contains out-of-range bus ids")

    scenario_labels = [str(column) for column in df.columns[1:]]
    if expected_steps is not None and len(scenario_labels) < expected_steps:
        raise ValueError(
            f"{path} {sheet_name} scenario count mismatch: expected at least {expected_steps}, got {len(scenario_labels)}"
        )
    if expected_steps is not None:
        scenario_labels = scenario_labels[:expected_steps]
        values = df.iloc[:, 1 : 1 + expected_steps].to_numpy(dtype=np.float64)
    else:
        values = df.iloc[:, 1:].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(f"{path} {sheet_name} contains NaN or infinite values")
    if (values < 0.0).any():
        raise ValueError(f"{path} {sheet_name} contains negative values")

    time_index = [_build_time_index(label, idx) for idx, label in enumerate(scenario_labels)]
    values = values * value_scale
    values_by_bus = {
        int(bus_id): values[row_index, :].astype(float).tolist()
        for row_index, bus_id in enumerate(bus_ids)
    }
    return ScenarioTable(
        name=table_name,
        bus_ids=[int(bus_id) for bus_id in bus_ids],
        time_index=time_index,
        values_by_bus=values_by_bus,
    )


def _split_pv_and_grid_tables(pv_raw: ScenarioTable, time_index: List[TimeIndex]) -> tuple[ScenarioTable, ScenarioTable]:
    zero_profile = [0.0] * len(time_index)
    pv_values: Dict[int, List[float]] = {}
    grid_values: Dict[int, List[float]] = {}

    for bus_id in range(1, NB + 1):
        raw_values = pv_raw.values_by_bus.get(bus_id, zero_profile)
        if bus_id in GRID_BUS_IDS:
            pv_values[bus_id] = zero_profile.copy()
            grid_values[bus_id] = list(raw_values)
        else:
            pv_values[bus_id] = list(raw_values)
            grid_values[bus_id] = zero_profile.copy()

    missing_grid = [bus_id for bus_id in GRID_BUS_IDS if max(grid_values[bus_id]) <= 0.0]
    if missing_grid:
        raise ValueError(f"PV scenario file does not contain positive grid capacity for buses: {missing_grid}")

    bus_ids = list(range(1, NB + 1))
    return (
        ScenarioTable("pv_p_max", bus_ids, time_index, pv_values),
        ScenarioTable("grid_p_max", bus_ids, time_index, grid_values),
    )


def _build_state_samples(tables: Dict[str, ScenarioTable]) -> List[StateSample192]:
    p_load = tables["p_load"]
    q_load = tables["q_load"]
    pv_p_max = tables["pv_p_max"]
    grid_p_max = tables["grid_p_max"]
    samples: List[StateSample192] = []

    for step_idx, time_info in enumerate(p_load.time_index):
        samples.append(
            StateSample192(
                time=time_info,
                p_load_by_bus={bus_id: p_load.value_at(bus_id, step_idx) for bus_id in p_load.bus_ids},
                q_load_by_bus={bus_id: q_load.value_at(bus_id, step_idx) for bus_id in q_load.bus_ids},
                pv_by_bus={bus_id: pv_p_max.value_at(bus_id, step_idx) for bus_id in pv_p_max.bus_ids},
                grid_p_max_by_bus={bus_id: grid_p_max.value_at(bus_id, step_idx) for bus_id in grid_p_max.bus_ids},
            )
        )
    return samples


def _validate_time_alignment(*tables: ScenarioTable) -> None:
    if not tables:
        raise ValueError("at least one table is required for alignment validation")
    reference_labels = [item.label for item in tables[0].time_index]
    for table in tables[1:]:
        labels = [item.label for item in table.time_index]
        if labels != reference_labels:
            raise ValueError(f"time labels of {tables[0].name} and {table.name} are not aligned")


def _build_time_index(label: str, fallback_index: int) -> TimeIndex:
    label = str(label)
    if label.upper().startswith("H"):
        try:
            global_step = int(label[1:])
        except ValueError:
            global_step = fallback_index
    else:
        global_step = fallback_index
    day_index = global_step // 24 + 1
    hour_of_day = global_step % 24 + 1
    return TimeIndex(label=label, global_step=global_step, day_index=day_index, hour_of_day=hour_of_day)


if __name__ == "__main__":
    dataset = load_distribution_dataset_192()
    sample = dataset.sample_at(0)
    print("192-bus dataset loaded")
    print(f"steps={dataset.num_steps}, buses={dataset.p_load.num_buses}")
    print(f"first={sample.time.label}, p_load_bus1={sample.p_load_by_bus[1]:.6f} MW")
