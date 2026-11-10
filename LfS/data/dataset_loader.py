"""
配电网重构任务的历史数据读取模块。

本模块不依赖 pandas/openpyxl, 而是直接解析 xlsx 文件内部的 XML 结构，
目的是在当前环境下稳定读取以下历史数据：

1. load scenario_33.xlsx
   - Sheet3: 节点有功负荷
   - Sheet4: 节点无功负荷

2. PV scenario_33.xlsx
   - 默认读取 Sheet3, 可通过参数修改

后续可直接基于本模块构造状态：
- 节点有功负荷
- 节点无功负荷
- 节点光伏出力
- 时间信息(连续时刻、天索引、1-24 小时索引）
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from xml.etree import ElementTree as ET


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


@dataclass(frozen=True)
class TimeIndex:
    """时间索引信息。

    global_step:
        原始标签中的绝对小时编号，例如 H24 -> 24。

    day_index:
        从 1 开始的天编号。这里按 24 小时一组切分。

    hour_of_day:
        1 到 24 的小时编号，便于后续状态建模。
    """

    label: str
    global_step: int
    day_index: int
    hour_of_day: int


@dataclass
class ScenarioTable:
    """单个场景表的统一表示。"""

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


@dataclass
class StateSample:
    """单个时刻的状态样本。"""

    time: TimeIndex
    p_load_by_bus: Dict[int, float]
    q_load_by_bus: Dict[int, float]
    pv_by_bus: Dict[int, float]


@dataclass
class DistributionDataset:
    """主程序友好的数据集封装。"""

    tables: Dict[str, ScenarioTable]
    samples: List[StateSample]

    @property
    def p_load(self) -> ScenarioTable:
        return self.tables["p_load"]

    @property
    def q_load(self) -> ScenarioTable:
        return self.tables["q_load"]

    @property
    def pv_output(self) -> ScenarioTable:
        return self.tables["pv_output"]

    @property
    def num_steps(self) -> int:
        return len(self.samples)

    def sample_at(self, step_idx: int) -> StateSample:
        """按时间步索引获取单个场景样本。"""
        return self.samples[step_idx]


class XlsxReader:
    """使用标准库读取 xlsx 的轻量级读取器。"""

    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"找不到 Excel 文件: {self.file_path}")

    def read_sheet_rows(self, sheet_name: str) -> List[List[str]]:
        """读取指定工作表，并返回按行组织的字符串矩阵。

        返回值约定：
        - 每一行都是等长列表
        - 空单元格会填成空字符串
        """

        with zipfile.ZipFile(self.file_path) as zf:
            shared_strings = self._read_shared_strings(zf)
            sheet_path = self._resolve_sheet_path(zf, sheet_name)
            root = ET.fromstring(zf.read(sheet_path))
            sheet_data = root.find(f"{{{NS_MAIN}}}sheetData")
            if sheet_data is None:
                raise ValueError(f"工作表 {sheet_name} 中未找到 sheetData")

            rows: List[List[str]] = []
            for row in sheet_data.findall(f"{{{NS_MAIN}}}row"):
                cell_map: Dict[int, str] = {}
                max_col_idx = -1

                for cell in row.findall(f"{{{NS_MAIN}}}c"):
                    ref = cell.attrib.get("r")
                    if ref is None:
                        raise ValueError(f"工作表 {sheet_name} 中存在缺少坐标引用的单元格")

                    col_idx = self._column_letters_to_index(self._extract_column_letters(ref))
                    value = self._read_cell_value(cell, shared_strings)
                    cell_map[col_idx] = value
                    max_col_idx = max(max_col_idx, col_idx)

                if max_col_idx < 0:
                    rows.append([])
                    continue

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
            text_parts: List[str] = []
            for text_node in si.iterfind(f".//{{{NS_MAIN}}}t"):
                text_parts.append(text_node.text or "")
            shared_strings.append("".join(text_parts))
        return shared_strings

    def _resolve_sheet_path(self, zf: zipfile.ZipFile, sheet_name: str) -> str:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = workbook.find(f"{{{NS_MAIN}}}sheets")
        if sheets is None:
            raise ValueError(f"文件 {self.file_path} 中未找到 sheets 定义")

        relationship_id: Optional[str] = None
        for sheet in sheets:
            if sheet.attrib.get("name") == sheet_name:
                relationship_id = sheet.attrib.get(f"{{{NS_REL}}}id")
                break

        if relationship_id is None:
            raise ValueError(f"文件 {self.file_path.name} 中未找到工作表 {sheet_name}")

        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        for rel in rels:
            if rel.attrib.get("Id") == relationship_id:
                target = rel.attrib["Target"].lstrip("/")
                if not target.startswith("xl/"):
                    target = f"xl/{target}"
                return target

        raise ValueError(f"工作表 {sheet_name} 的关系 id {relationship_id} 无法解析")

    @staticmethod
    def _extract_column_letters(cell_ref: str) -> str:
        match = re.match(r"([A-Z]+)", cell_ref)
        if match is None:
            raise ValueError(f"无法解析单元格列引用: {cell_ref}")
        return match.group(1)

    @staticmethod
    def _column_letters_to_index(column_letters: str) -> int:
        index = 0
        for char in column_letters:
            index = index * 26 + (ord(char) - ord("A") + 1)
        return index - 1

    @staticmethod
    def _read_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> str:
        cell_type = cell.attrib.get("t")
        if cell_type == "inlineStr":
            text_node = cell.find(f".//{{{NS_MAIN}}}t")
            return "" if text_node is None else (text_node.text or "")

        value_node = cell.find(f"{{{NS_MAIN}}}v")
        if value_node is None:
            return ""

        raw_value = value_node.text or ""
        if cell_type == "s":
            shared_idx = int(raw_value)
            return shared_strings[shared_idx]

        return raw_value


def _parse_time_label(label: str, step_idx: int) -> TimeIndex:
    """将表头时间标签转换为从数据首列开始的连续时间索引。

    原始 Excel 表头可能从 H24 开始，但训练样本的时间语义应以当前
    数据集第一列作为 day1 hour1，因此 day/hour 按列序号计算。
    """

    match = re.fullmatch(r"H(\d+)", label.strip())
    if match is None:
        raise ValueError(f"时间标签格式不合法，预期类似 H24，实际得到: {label}")

    raw_global_step = int(match.group(1))
    if raw_global_step <= 0:
        raise ValueError(f"时间步必须为正整数，实际得到: {label}")

    if step_idx < 0:
        raise ValueError(f"step_idx must be non-negative, got {step_idx}")

    global_step = step_idx + 1
    # 这里按数据列序号而不是 H 标签数值计算时间，保证首个样本为 day1 hour1。
    day_index = (global_step - 1) // 24 + 1
    hour_of_day = (global_step - 1) % 24 + 1

    return TimeIndex(
        label=label,
        global_step=global_step,
        day_index=day_index,
        hour_of_day=hour_of_day,
    )


def _to_int(value: str, field_name: str) -> int:
    if value == "":
        raise ValueError(f"{field_name} 为空，无法转换为整数")
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} 无法转换为整数: {value}") from exc


def _to_float(value: str, field_name: str) -> float:
    if value == "":
        raise ValueError(f"{field_name} 为空，无法转换为浮点数")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} 无法转换为浮点数: {value}") from exc


def parse_scenario_sheet(
    file_path: Path | str,
    sheet_name: str,
    table_name: str,
) -> ScenarioTable:
    """将一个历史场景工作表解析成统一表结构。"""

    reader = XlsxReader(file_path)
    rows = reader.read_sheet_rows(sheet_name)
    if len(rows) < 2:
        raise ValueError(f"{file_path} 的 {sheet_name} 至少需要包含表头和一行数据")

    header = rows[0]
    if len(header) < 2:
        raise ValueError(f"{file_path} 的 {sheet_name} 表头列数不足")

    first_col_name = header[0].strip().lower()
    if first_col_name not in {"bus", "node", "bus_id", "node_id"}:
        raise ValueError(
            f"{file_path} 的 {sheet_name} 第一列表头应表示节点编号，实际得到: {header[0]}"
        )

    time_labels = header[1:]
    time_index = [_parse_time_label(label, step_idx) for step_idx, label in enumerate(time_labels)]

    bus_ids: List[int] = []
    values_by_bus: Dict[int, List[float]] = {}

    expected_num_steps = len(time_index)

    for row_idx, row in enumerate(rows[1:], start=2):
        if not row:
            continue

        bus_id = _to_int(row[0], f"{sheet_name} 第 {row_idx} 行节点编号")
        raw_values = row[1:]

        if len(raw_values) < expected_num_steps:
            raise ValueError(
                f"{sheet_name} 第 {row_idx} 行数据列不足，"
                f"预期 {expected_num_steps} 列，实际 {len(raw_values)} 列"
            )

        values = [
            _to_float(raw_values[i], f"{sheet_name} 第 {row_idx} 行第 {i + 2} 列")
            for i in range(expected_num_steps)
        ]

        if bus_id in values_by_bus:
            raise ValueError(f"{sheet_name} 中节点编号重复: {bus_id}")

        bus_ids.append(bus_id)
        values_by_bus[bus_id] = values

    return ScenarioTable(
        name=table_name,
        bus_ids=bus_ids,
        time_index=time_index,
        values_by_bus=values_by_bus,
    )


def load_distribution_reconfiguration_data(
    base_dir: Path | str,
    load_file_name: str = "load scenario_33.xlsx",
    pv_file_name: str = "PV scenario_33.xlsx",
    p_sheet_name: str = "Sheet3",
    q_sheet_name: str = "Sheet4",
    pv_sheet_name: str = "Sheet3",
) -> Dict[str, ScenarioTable]:
    """读取配电网重构任务所需的历史场景数据。"""

    base_dir = Path(base_dir)
    load_file = base_dir / load_file_name
    pv_file = base_dir / pv_file_name

    p_load = parse_scenario_sheet(load_file, p_sheet_name, "p_load")
    q_load = parse_scenario_sheet(load_file, q_sheet_name, "q_load")
    pv_output = parse_scenario_sheet(pv_file, pv_sheet_name, "pv_output")

    _validate_tables_can_align(p_load, q_load, pv_output)

    return {
        "p_load": p_load,
        "q_load": q_load,
        "pv_output": pv_output,
    }


def build_state_samples(tables: Dict[str, ScenarioTable]) -> List[StateSample]:
    """将场景表转换为按时间组织的状态样本列表。"""

    p_load = tables["p_load"]
    q_load = tables["q_load"]
    pv_output = tables["pv_output"]

    samples: List[StateSample] = []
    for step_idx, time_info in enumerate(p_load.time_index):
        p_values = {bus_id: p_load.value_at(bus_id, step_idx) for bus_id in p_load.bus_ids}
        q_values = {bus_id: q_load.value_at(bus_id, step_idx) for bus_id in q_load.bus_ids}
        pv_values = {bus_id: pv_output.value_at(bus_id, step_idx) for bus_id in pv_output.bus_ids}

        samples.append(
            StateSample(
                time=time_info,
                p_load_by_bus=p_values,
                q_load_by_bus=q_values,
                pv_by_bus=pv_values,
            )
        )

    return samples


def get_default_project_root() -> Path:
    """返回项目根目录，便于主程序直接复用默认数据位置。"""

    return Path(__file__).resolve().parents[1]


def load_distribution_dataset(
    base_dir: Path | str | None = None,
    load_file_name: str = "load scenario_33.xlsx",
    pv_file_name: str = "PV scenario_33.xlsx",
    p_sheet_name: str = "Sheet3",
    q_sheet_name: str = "Sheet4",
    pv_sheet_name: str = "Sheet3",
) -> DistributionDataset:
    """主程序推荐调用的统一入口。

    返回值同时包含：
    - tables: 原始场景表
    - samples: 按时间步构造好的状态样本
    """

    if base_dir is None:
        base_dir = get_default_project_root()

    tables = load_distribution_reconfiguration_data(
        base_dir=base_dir,
        load_file_name=load_file_name,
        pv_file_name=pv_file_name,
        p_sheet_name=p_sheet_name,
        q_sheet_name=q_sheet_name,
        pv_sheet_name=pv_sheet_name,
    )
    samples = build_state_samples(tables)
    return DistributionDataset(tables=tables, samples=samples)


def get_state_sample(
    step_idx: int,
    base_dir: Path | str | None = None,
) -> StateSample:
    """为主程序提供的便捷接口: 直接取某个时间步样本。"""

    dataset = load_distribution_dataset(base_dir=base_dir)
    return dataset.sample_at(step_idx)


def _validate_tables_can_align(*tables: ScenarioTable) -> None:
    """校验多个场景表是否可以按时间对齐。"""

    if not tables:
        raise ValueError("至少需要一个场景表进行对齐校验")

    ref = tables[0]
    ref_labels = [item.label for item in ref.time_index]

    for table in tables[1:]:
        labels = [item.label for item in table.time_index]
        if labels != ref_labels:
            raise ValueError(
                f"时间轴不一致，{ref.name} 与 {table.name} 无法对齐"
            )


def _print_sample_preview(samples: Sequence[StateSample], preview_count: int = 24) -> None:
    """打印前若干个场景样本，便于人工核对读取结果。"""

    preview_count = min(preview_count, len(samples))
    print(f"\n前 {preview_count} 个场景预览:")
    for idx, sample in enumerate(samples[:preview_count], start=1):
        print(
            f"[scenario {idx:02d}] "
            f"time={sample.time.label}, "
            f"day={sample.time.day_index}, "
            f"hour={sample.time.hour_of_day}"
        )
        print(f"  p_load[:5] = {list(sample.p_load_by_bus.items())[:5]}")
        print(f"  q_load[:5] = {list(sample.q_load_by_bus.items())[:5]}")
        print(f"  pv[:5] = {list(sample.pv_by_bus.items())[:5]}")


# Main-program call examples:
# from data.dataset_loader import load_distribution_dataset, get_state_sample
#
# dataset = load_distribution_dataset()
# print(dataset.num_steps)
# first_sample = dataset.sample_at(0)
#
# sample = get_state_sample(step_idx=0)
#
# p_load_table = dataset.p_load
# q_load_table = dataset.q_load
# pv_table = dataset.pv_output
#
if __name__ == "__main__":
    # 这里提供最小可运行示例，方便后续直接检查读取结果是否正确。
    project_root = Path(__file__).resolve().parents[1]
    dataset = load_distribution_dataset()
    tables = dataset.tables
    samples = dataset.samples

    print("历史数据读取完成")
    print(f"有功负荷节点数: {tables['p_load'].num_buses}")
    print(f"无功负荷节点数: {tables['q_load'].num_buses}")
    print(f"光伏节点数: {tables['pv_output'].num_buses}")
    print(f"时间步数: {tables['p_load'].num_steps}")

    first_sample = samples[0]
    print(
        f"首个样本时间: {first_sample.time.label}, "
        f"day={first_sample.time.day_index}, "
        f"hour={first_sample.time.hour_of_day}"
    )
    print(f"首个样本前 5 个有功节点: {list(first_sample.p_load_by_bus.items())[:5]}")
    print(f"首个样本前 5 个无功节点: {list(first_sample.q_load_by_bus.items())[:5]}")
    print(f"首个样本前 5 个光伏节点: {list(first_sample.pv_by_bus.items())[:5]}")
    _print_sample_preview(samples, preview_count=24)
