from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from agent.topology_repair_192 import repair_scores_to_radial_topology
from agent.topology_validator_192 import TopologyValidationResult, validate_radial_topology


@dataclass(frozen=True)
class DecodedTopology:
    topology: np.ndarray
    action_scores: np.ndarray
    validation: TopologyValidationResult
    decode_mode: str


def normalize_action_scores(action_raw: Sequence[float]) -> np.ndarray:
    """将 actor 输出转为 [0, 1] 支路优先级分数，便于统一修复和排查。"""

    action = np.asarray(action_raw, dtype=np.float32).reshape(-1)
    if np.any(~np.isfinite(action)):
        raise ValueError("action_raw contains NaN or infinite values")

    if np.min(action) < 0.0 or np.max(action) > 1.0:
        return np.clip((action + 1.0) * 0.5, 0.0, 1.0)
    return np.clip(action, 0.0, 1.0)


def decode_topology_scores(
    action_raw: Sequence[float],
    branch_data: Sequence[Dict[str, float]],
    num_buses: int,
) -> DecodedTopology:
    """把连续动作解码成 192 节点双电源径向森林拓扑。"""

    scores = normalize_action_scores(action_raw)
    repair_result = repair_scores_to_radial_topology(scores, branch_data, num_buses)
    validation = validate_radial_topology(repair_result.topology, branch_data, num_buses)
    if not validation.is_feasible:
        raise RuntimeError(
            "decoded topology is still infeasible after repair: "
            f"reasons={validation.reasons}"
        )

    return DecodedTopology(
        topology=repair_result.topology,
        action_scores=scores,
        validation=validation,
        decode_mode=repair_result.repair_mode,
    )
