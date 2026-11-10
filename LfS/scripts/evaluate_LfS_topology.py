from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.LfS_topology import LfSTopology
from agent.replay_memory import DiffusionMemory, ReplayMemory
from data.dataset_loader import load_distribution_dataset
from env_setting.socp_33 import NB, NL
from scripts.train_LfS_topology import (
    _build_agent_args,
    _import_torch,
    _load_actor_if_provided,
    _load_critic_if_provided,
    evaluate_LfS_topology_agent,
)


def _agent_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        policy_type=args.policy_type,
        critic_type=args.critic_type,
        noise_ratio=args.noise_ratio,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,
        behavior_sample=args.behavior_sample,
        eval_sample=args.eval_sample,
        deterministic=args.deterministic,
        target_sample=args.target_sample,
        diffusion_lr=1e-4,
        critic_lr=3e-4,
        tau=0.005,
        policy_freq=1,
        ac_grad_norm=2.0,
        update_actor_target_every=1,
        topology_candidate_count=args.eval_candidate_count,
        critic_hidden_dim=args.critic_hidden_dim,
        categorical_max_corruption=args.categorical_max_corruption,
        categorical_temperature=args.categorical_temperature,
        categorical_hidden_dim=args.categorical_hidden_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    torch = _import_torch()
    device = torch.device(args.device)
    replay_memory = ReplayMemory(args.state_dim, NL, capacity=1, device=device)
    diffusion_memory = DiffusionMemory(args.state_dim, NL, capacity=1, device=device)
    agent = LfSTopology(
        args=_agent_args(args),
        state_dim=args.state_dim,
        memory=replay_memory,
        diffusion_memory=diffusion_memory,
        device=device,
    )
    _load_actor_if_provided(agent, args.actor_path, torch)
    _load_critic_if_provided(agent, args.critic_path, torch)

    if args.eval_state_source == "historical":
        historical_dataset = load_distribution_dataset(base_dir=PROJECT_ROOT)
    elif args.eval_state_source == "suboptimal":
        historical_dataset = load_distribution_dataset(
            base_dir=PROJECT_ROOT,
            load_file_name=args.suboptimal_state_file_name,
            pv_file_name=args.suboptimal_state_file_name,
            p_sheet_name=args.suboptimal_p_load_sheet_name,
            q_sheet_name=args.suboptimal_q_load_sheet_name,
            pv_sheet_name=args.suboptimal_pv_p_sheet_name,
        )
    else:
        historical_dataset = None
    payload = evaluate_LfS_topology_agent(
        agent=agent,
        args=args,
        scenario_start=args.eval_scenario_start,
        scenario_count=args.eval_scenario_count,
        state_source=args.eval_state_source,
        historical_dataset=historical_dataset,
        label=args.label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained topology LfS actor+critic on a frozen validation set.")
    # 评估脚本默认跟随当前 CategoricalDiffusion actor + GNN critic 训练产物；后续若模型结构或训练目录变化，
    # 需要同步检查 checkpoint 路径、policy_type、critic_type 和安全 reward 参数，避免评估对象错配。
    parser.add_argument(
        "--actor-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "LfS_topology_training_categorical_gnn_critic_33" / "best_eval_actor.pth",
    )
    parser.add_argument(
        "--critic-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "LfS_topology_training_categorical_gnn_critic_33" / "best_eval_critic.pth",
    )
    parser.add_argument("--eval-state-source", choices=["env_testing", "historical", "suboptimal"], default="suboptimal")
    parser.add_argument("--eval-scenario-start", type=int, default=3600)
    parser.add_argument("--eval-scenario-count", type=int, default=720)
    parser.add_argument(
        "--suboptimal-state-file-name",
        type=str,
        default="data/suboptimal_33/suboptimal_dataset_33_debug.xlsx",
    )
    parser.add_argument("--suboptimal-p-load-sheet-name", type=str, default="Active_Load_MW")
    parser.add_argument("--suboptimal-q-load-sheet-name", type=str, default="Reactive_Load_MVar")
    parser.add_argument("--suboptimal-pv-p-sheet-name", type=str, default="PV_P_Max_MW")
    parser.add_argument("--eval-candidate-count", type=int, default=64)
    parser.add_argument("--state-dim", type=int, choices=[NB * 3, NB * 4], default=NB * 3)
    parser.add_argument("--power-scale", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=["negative_cost", "cost"], default="negative_cost")
    parser.add_argument("--invalid-solution-cost", type=float, default=1e8)
    parser.add_argument("--solver-time-limit", type=int, default=30)
    parser.add_argument("--voltage-penalty-lambda", type=float, default=10000.0)
    parser.add_argument("--current-penalty-lambda", type=float, default=10000.0)
    parser.add_argument("--voltage-deadband", type=float, default=0.05)
    parser.add_argument("--current-margin-ratio", type=float, default=0.9)
    parser.add_argument("--drop-infeasible-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-reward", type=float, default=-10000.0)
    parser.add_argument("--max-reward", type=float, default=None)
    parser.add_argument("--min-candidates-after-filter", type=int, default=2)
    parser.add_argument("--noise-ratio", type=float, default=1.0)
    parser.add_argument(
        "--policy-type",
        choices=["Diffusion", "CategoricalDiffusion", "GNNCategoricalDiffusion"],
        default="CategoricalDiffusion",
    )
    parser.add_argument("--critic-type", choices=["MLP", "GNN"], default="GNN")
    parser.add_argument("--beta-schedule", choices=["linear", "cosine", "vp"], default="vp")
    parser.add_argument("--n-timesteps", type=int, default=20)
    parser.add_argument("--categorical-max-corruption", type=float, default=0.5)
    parser.add_argument("--categorical-temperature", type=float, default=1.0)
    parser.add_argument("--categorical-hidden-dim", type=int, default=256)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--behavior-sample", type=int, default=64)
    parser.add_argument("--eval-sample", type=int, default=64)
    parser.add_argument("--target-sample", type=int, default=64)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--label", type=str, default="validation")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "LfS_topology_training_categorical_gnn_critic_33" / "LfS_topology_eval_report.json",
    )
    args = parser.parse_args()
    if args.eval_scenario_count <= 0:
        raise ValueError(f"eval-scenario-count must be positive, got {args.eval_scenario_count}")
    if args.eval_candidate_count <= 0:
        raise ValueError(f"eval-candidate-count must be positive, got {args.eval_candidate_count}")
    if not 0.0 <= args.categorical_max_corruption <= 1.0:
        raise ValueError(
            f"categorical-max-corruption must be in [0, 1], got {args.categorical_max_corruption}"
        )
    if args.categorical_temperature <= 0.0:
        raise ValueError(f"categorical-temperature must be positive, got {args.categorical_temperature}")
    if args.categorical_hidden_dim <= 0:
        raise ValueError(f"categorical-hidden-dim must be positive, got {args.categorical_hidden_dim}")
    if args.gnn_hidden_dim <= 0:
        raise ValueError(f"gnn-hidden-dim must be positive, got {args.gnn_hidden_dim}")
    if args.gnn_layers <= 0:
        raise ValueError(f"gnn-layers must be positive, got {args.gnn_layers}")
    if args.voltage_penalty_lambda < 0.0:
        raise ValueError(f"voltage-penalty-lambda must be non-negative, got {args.voltage_penalty_lambda}")
    if args.current_penalty_lambda < 0.0:
        raise ValueError(f"current-penalty-lambda must be non-negative, got {args.current_penalty_lambda}")
    if args.voltage_deadband < 0.0:
        raise ValueError(f"voltage-deadband must be non-negative, got {args.voltage_deadband}")
    if not 0.0 <= args.current_margin_ratio <= 1.0:
        raise ValueError(f"current-margin-ratio must be in [0, 1], got {args.current_margin_ratio}")
    if args.min_candidates_after_filter <= 0:
        raise ValueError(
            f"min-candidates-after-filter must be positive, got {args.min_candidates_after_filter}"
        )
    if args.min_reward is not None and args.max_reward is not None and args.min_reward > args.max_reward:
        raise ValueError(f"min-reward cannot be greater than max-reward: {args.min_reward} > {args.max_reward}")
    return args


def main() -> None:
    payload = run_evaluation(parse_args())
    metrics = payload["metrics"]
    print("topology LfS evaluation complete")
    print(f"scenario range: {metrics['scenario_start']} - {metrics['scenario_start'] + metrics['scenario_count'] - 1}")
    print(f"evaluated_scenario_count: {metrics['evaluated_scenario_count']}")
    print(f"critic_top1_accuracy: {metrics['critic_top1_accuracy']:.4f}")
    print(f"critic_pairwise_order_accuracy: {metrics['critic_pairwise_order_accuracy']:.4f}")
    print(f"critic_mean_regret: {metrics['critic_mean_regret']:.6f}")
    print(f"critic_p95_regret: {metrics['critic_p95_regret']:.6f}")
    print(f"candidate_count_unique_mean: {metrics['candidate_count_unique_mean']:.2f}")
    print(f"candidate_feasible_ratio: {metrics['candidate_feasible_ratio']:.4f}")


if __name__ == "__main__":
    main()
