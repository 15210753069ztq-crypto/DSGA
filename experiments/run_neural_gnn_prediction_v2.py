from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_neural_gnn_prediction import (
    ACTIONS,
    EXPECTED_ACTION,
    SCENARIOS,
    build_methods,
    dependency_score,
    evaluate,
    make_windows,
    marginal_score,
    pairs,
    ridge_estimate,
    set_seed,
    summarize,
)


class DecoupledSelectiveGraphForecaster(nn.Module):
    """Forecasting branches whose adapters cannot directly modify each other."""

    def __init__(
        self,
        n_nodes: int,
        lag: int,
        support: np.ndarray,
        hidden: int = 24,
        graph_rank: int = 8,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.temporal = nn.Sequential(
            nn.Linear(lag, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        prior = torch.as_tensor(support, dtype=torch.float32)
        logits = torch.where(
            prior > 0,
            torch.full_like(prior, 2.2),
            torch.full_like(prior, -2.2),
        )
        logits.fill_diagonal_(-8.0)
        self.base_graph_logits = nn.Parameter(logits)
        self.register_buffer("off_diagonal", 1.0 - torch.eye(n_nodes))

        self.temporal_readout = nn.Linear(hidden, 1)
        self.graph_readout = nn.Linear(hidden, 1, bias=False)
        self.time_scale = nn.Parameter(torch.zeros(n_nodes))
        self.time_bias = nn.Parameter(torch.zeros(n_nodes))

        generator = torch.Generator().manual_seed(12345)
        self.graph_u = nn.Parameter(
            torch.randn(n_nodes, graph_rank, generator=generator) * 0.5
        )
        self.graph_v = nn.Parameter(torch.zeros(n_nodes, graph_rank))

    def adjacency(self) -> torch.Tensor:
        delta = 5.0 * (self.graph_u @ self.graph_v.T)
        adjacency = torch.sigmoid(self.base_graph_logits + delta) * self.off_diagonal
        return adjacency / (adjacency.sum(dim=1, keepdim=True) + 1e-6)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        hidden = self.temporal(history)
        local_prediction = self.temporal_readout(hidden).squeeze(-1)
        messages = torch.einsum("ij,bjh->bih", self.adjacency(), hidden)
        graph_prediction = self.graph_readout(messages).squeeze(-1)
        time_correction = local_prediction * self.time_scale + self.time_bias
        return local_prediction + graph_prediction + time_correction


def configure_action(model: DecoupledSelectiveGraphForecaster, action: str) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if action == "base":
        for parameter in model.temporal.parameters():
            parameter.requires_grad = True
        model.base_graph_logits.requires_grad = True
        for parameter in model.temporal_readout.parameters():
            parameter.requires_grad = True
        for parameter in model.graph_readout.parameters():
            parameter.requires_grad = True
    elif action == "time":
        model.time_scale.requires_grad = True
        model.time_bias.requires_grad = True
    elif action == "graph":
        model.graph_u.requires_grad = True
        model.graph_v.requires_grad = True
    elif action == "joint":
        model.time_scale.requires_grad = True
        model.time_bias.requires_grad = True
        model.graph_u.requires_grad = True
        model.graph_v.requires_grad = True
    elif action == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif action != "freeze":
        raise ValueError(action)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit(
    model: DecoupledSelectiveGraphForecaster,
    train_set: TensorDataset,
    validation_set: TensorDataset,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> DecoupledSelectiveGraphForecaster:
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        return model
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)
    loader = DataLoader(train_set, batch_size=256, shuffle=True)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale = 0
    for _ in range(epochs):
        model.train()
        for history, target in loader:
            history = history.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history)
            forecast_loss = torch.mean((prediction - target) ** 2)
            graph_penalty = torch.mean(model.graph_v**2)
            time_penalty = torch.mean(model.time_scale**2) + torch.mean(
                model.time_bias**2
            )
            loss = forecast_loss + 2e-5 * graph_penalty + 2e-5 * time_penalty
            loss.backward()
            optimizer.step()
        validation, _ = evaluate(model, validation_set, device)
        if validation < best_validation - 1e-7:
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    model.load_state_dict(best_state)
    return model


def calibrate_null(
    base_values: np.ndarray,
    seed: int,
    window: int,
    null_windows: int,
    alpha: float,
) -> dict[str, object]:
    base_x, base_y = pairs(base_values[:3500])
    base_parameters = ridge_estimate(base_x, base_y, ridge=1.0)
    reference = base_values[3500 - window : 3500]
    rng = np.random.default_rng(10_000 + seed)
    starts = rng.integers(4500, 5500 - window + 1, size=null_windows)
    marginal_null = np.asarray(
        [marginal_score(reference, base_values[s : s + window]) for s in starts]
    )
    dependency_null = np.asarray(
        [dependency_score(base_values[s : s + window], base_parameters) for s in starts]
    )
    return {
        "reference": reference,
        "base_parameters": base_parameters,
        "marginal_null": marginal_null,
        "dependency_null": dependency_null,
        "threshold_marginal": float(np.quantile(marginal_null, 1.0 - alpha)),
        "threshold_dependency": float(np.quantile(dependency_null, 1.0 - alpha)),
    }


def calibrated_route(
    current: np.ndarray,
    calibration: dict[str, object],
    window: int,
    alpha: float,
    uncertain_alpha: float,
    minimum_effect_z: float = 2.5,
) -> tuple[str, float, float, float, float, bool]:
    marginal = marginal_score(calibration["reference"], current[:window])
    dependency = dependency_score(
        current[:window], calibration["base_parameters"]
    )
    marginal_null = calibration["marginal_null"]
    dependency_null = calibration["dependency_null"]
    p_marginal = float((1 + np.sum(marginal_null >= marginal)) / (len(marginal_null) + 1))
    p_dependency = float(
        (1 + np.sum(dependency_null >= dependency)) / (len(dependency_null) + 1)
    )
    def robust_z(value: float, null: np.ndarray) -> float:
        center = float(np.median(null))
        mad = float(np.median(np.abs(null - center)))
        scale = max(1.4826 * mad, float(np.std(null, ddof=1)) * 0.25, 1e-8)
        return (value - center) / scale

    z_marginal = robust_z(marginal, marginal_null)
    z_dependency = robust_z(dependency, dependency_null)
    marginal_flag = p_marginal <= alpha and z_marginal >= minimum_effect_z
    dependency_flag = p_dependency <= alpha and z_dependency >= minimum_effect_z
    if marginal_flag and dependency_flag:
        action = "joint"
    elif marginal_flag:
        action = "time"
    elif dependency_flag:
        action = "graph"
    else:
        action = "freeze"
    uncertain = (
        (not marginal_flag and p_marginal <= uncertain_alpha and z_marginal >= 1.5)
        or (
            not dependency_flag
            and p_dependency <= uncertain_alpha
            and z_dependency >= 1.5
        )
    )
    return action, marginal, dependency, p_marginal, p_dependency, uncertain


def action_learning_rate(action: str) -> float:
    if action == "graph":
        return 1e-2
    if action == "joint":
        return 6e-3
    if action == "time":
        return 3e-3
    return 8e-4


def run_seed(
    dataset_path: Path,
    seed: int,
    device: torch.device,
    base_epochs: int,
    adapt_epochs: int,
    adaptation_points: int,
    lag: int,
    null_windows: int,
    alpha: float,
    uncertain_alpha: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    payload = np.load(dataset_path)
    base_values = payload["base"]
    support = payload["base_support"].astype(bool)
    base_train = make_windows(base_values, 0, 3500, lag)
    base_validation = make_windows(base_values, 3500, 4500, lag)
    base_test = make_windows(base_values, 5500, 6000, lag)
    calibration = calibrate_null(
        base_values,
        seed,
        adaptation_points,
        null_windows,
        alpha,
    )

    set_seed(seed)
    model = DecoupledSelectiveGraphForecaster(
        base_values.shape[1], lag, support
    ).to(device)
    configure_action(model, "base")
    model = fit(model, base_train, base_validation, device, base_epochs, 2e-3)
    base_mse, base_mae = evaluate(model, base_test, device)
    full_parameters = sum(p.numel() for p in model.parameters())

    action_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        values = payload[scenario]
        adaptation_set = make_windows(values, 0, adaptation_points, lag)
        validation_set = make_windows(values, adaptation_points, 1024, lag)
        current_test = make_windows(values, 2000, 6000, lag)
        (
            routed_action,
            score_marginal,
            score_dependency,
            p_marginal,
            p_dependency,
            uncertain,
        ) = calibrated_route(
            values,
            calibration,
            adaptation_points,
            alpha,
            uncertain_alpha,
        )
        routing_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "routed_action": routed_action,
                "expected_action": EXPECTED_ACTION[scenario],
                "score_marginal": score_marginal,
                "score_dependency": score_dependency,
                "p_marginal": p_marginal,
                "p_dependency": p_dependency,
                "uncertain": uncertain,
                "threshold_marginal": calibration["threshold_marginal"],
                "threshold_dependency": calibration["threshold_dependency"],
            }
        )
        for action in ACTIONS:
            set_seed(seed * 100 + ACTIONS.index(action))
            adapted = copy.deepcopy(model).to(device)
            trainable = configure_action(adapted, action)
            started = time.perf_counter()
            if action != "freeze" and trainable > 0:
                adapted = fit(
                    adapted,
                    adaptation_set,
                    validation_set,
                    device,
                    adapt_epochs,
                    action_learning_rate(action),
                )
            elapsed = time.perf_counter() - started
            validation_mse, validation_mae = evaluate(
                adapted, validation_set, device
            )
            current_mse, current_mae = evaluate(adapted, current_test, device)
            history_mse, history_mae = evaluate(adapted, base_test, device)
            forgetting = max(history_mse - base_mse, 0.0)
            utility = current_mse + 0.25 * forgetting + 0.002 * trainable / full_parameters
            action_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "action": action,
                    "expected_action": EXPECTED_ACTION[scenario],
                    "current_mse": current_mse,
                    "current_mae": current_mae,
                    "validation_mse": validation_mse,
                    "validation_mae": validation_mae,
                    "history_mse": history_mse,
                    "history_mae": history_mae,
                    "base_mse": base_mse,
                    "base_mae": base_mae,
                    "forgetting": forgetting,
                    "utility": utility,
                    "trainable_parameters": trainable,
                    "trainable_fraction": trainable / full_parameters,
                    "adaptation_seconds": elapsed,
                }
            )
    return action_rows, routing_rows


def build_guarded_router(
    actions: pd.DataFrame,
    routing: pd.DataFrame,
    candidate_alpha: float,
) -> pd.DataFrame:
    merged = actions.merge(routing, on=["seed", "scenario", "expected_action"])
    selected = []
    for (_, _), group in merged.groupby(["seed", "scenario"]):
        route_row = group.iloc[0]
        routed = str(route_row.routed_action)
        candidates = {"freeze", routed}
        if routed == "joint":
            candidates.update({"time", "graph"})
        elif routed == "time" and route_row.p_dependency <= candidate_alpha:
            candidates.add("joint")
        elif routed == "graph" and route_row.p_marginal <= candidate_alpha:
            candidates.add("joint")
        elif routed == "freeze":
            if route_row.p_marginal <= candidate_alpha:
                candidates.add("time")
            if route_row.p_dependency <= candidate_alpha:
                candidates.add("graph")
        candidate_rows = group[group.action.isin(candidates)].copy()
        candidate_rows["selection_score"] = (
            candidate_rows.validation_mse
            + 0.002 * candidate_rows.trainable_fraction
        )
        choice = candidate_rows.loc[candidate_rows.selection_score.idxmin()].copy()
        choice["method"] = "guarded_router"
        choice["candidate_actions"] = ",".join(sorted(candidates))
        selected.append(choice)
    return pd.DataFrame(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-epochs", type=int, default=35)
    parser.add_argument("--adapt-epochs", type=int, default=15)
    parser.add_argument("--adaptation-points", type=int, default=512)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--null-windows", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--uncertain-alpha", type=float, default=0.10)
    parser.add_argument("--candidate-alpha", type=float, default=0.25)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    action_rows = []
    routing_rows = []
    for seed in range(args.seeds):
        print(f"Running SCA-GNN v2 seed {seed} on {device}", flush=True)
        seed_actions, seed_routing = run_seed(
            args.data / f"seed_{seed:02d}.npz",
            seed,
            device,
            args.base_epochs,
            args.adapt_epochs,
            args.adaptation_points,
            args.lag,
            args.null_windows,
            args.alpha,
            args.uncertain_alpha,
        )
        action_rows.extend(seed_actions)
        routing_rows.extend(seed_routing)

    actions = pd.DataFrame(action_rows)
    routing = pd.DataFrame(routing_rows)
    methods = build_methods(actions, routing)
    guarded = build_guarded_router(actions, routing, args.candidate_alpha)
    methods = pd.concat((methods, guarded), ignore_index=True)
    assessment = summarize(methods, routing)
    assessment["method_version"] = "domain-calibrated_decoupled-v2"
    assessment["uncertain_fraction"] = float(routing.uncertain.mean())
    overall = pd.DataFrame(assessment["overall"])
    guarded_utility = float(
        overall.loc[overall.method == "guarded_router", "utility"].iloc[0]
    )
    fixed = overall[overall.method.str.startswith("always_")].sort_values("utility")
    assessment["guarded_minus_best_fixed_utility"] = (
        guarded_utility - float(fixed.iloc[0].utility)
    )
    actions.to_csv(args.output / "neural_action_results.csv", index=False)
    routing.to_csv(args.output / "neural_routing_decisions.csv", index=False)
    methods.to_csv(args.output / "neural_method_results.csv", index=False)
    (args.output / "neural_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    print(json.dumps(assessment, indent=2), flush=True)


if __name__ == "__main__":
    main()
