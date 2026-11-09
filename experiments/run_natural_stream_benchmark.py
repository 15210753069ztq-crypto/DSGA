from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from run_neural_gnn_prediction import (
    dependency_drift_score,
    dependency_score,
    dependency_score_cross,
    evaluate,
    fit_temporal_matched_ar,
    fit_warmup_cross_model,
    make_windows,
    marginal_score,
    pairs,
    ridge_estimate,
    set_seed,
)
from run_neural_gnn_prediction_v2 import (
    DecoupledSelectiveGraphForecaster,
    action_learning_rate,
    configure_action,
    fit,
)


FIXED_ACTIONS = ("freeze", "time", "graph", "joint", "full")
METHODS = tuple(f"always_{action}" for action in FIXED_ACTIONS) + (
    "experience_replay",
    "sca_router",
    "guarded_router",
    "ewc",
    "lwf",
)


def load_standardized_csv(path: Path, warmup_ratio: float) -> tuple[np.ndarray, dict]:
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError(f"No numeric columns found in {path}")
    values = numeric.to_numpy(dtype=np.float32)
    warmup_end = int(len(values) * warmup_ratio)
    if warmup_end < 1000:
        raise ValueError("Warm-up phase is too short for calibration and base training")
    mean = values[:warmup_end].mean(axis=0)
    std = values[:warmup_end].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    standardized = (values - mean) / std
    metadata = {
        "rows": int(len(values)),
        "numeric_channels": int(values.shape[1]),
        "warmup_end": warmup_end,
        "normalization": "mean/std estimated only from warm-up phase",
        "numeric_columns": numeric.columns.tolist(),
    }
    return standardized.astype(np.float32), metadata


def infer_support(values: np.ndarray, top_k: int) -> np.ndarray:
    differences = np.diff(values, axis=0)
    correlation = np.nan_to_num(np.corrcoef(differences, rowvar=False))
    n_nodes = correlation.shape[0]
    support = np.zeros((n_nodes, n_nodes), dtype=bool)
    neighbors = min(top_k, n_nodes - 1)
    for target in range(n_nodes):
        scores = np.abs(correlation[target]).copy()
        scores[target] = -np.inf
        selected = np.argpartition(scores, -neighbors)[-neighbors:]
        support[target, selected] = True
    np.fill_diagonal(support, False)
    return support


def calibrate_natural_null(
    values: np.ndarray,
    train_end: int,
    lag: int,
    window: int,
    null_windows: int,
    alpha: float,
    seed: int,
    support: np.ndarray,
    diag_lag: int,
) -> dict[str, object]:
    base_x, base_y = pairs(values[:train_end])
    base_parameters = ridge_estimate(base_x, base_y, ridge=1.0)
    # 每变量多阶自回归基线（temporal-matched），用于生成诊断残差
    base_ar = fit_temporal_matched_ar(values[:train_end], diag_lag)
    # warm-up 跨变量残差模型快照 + warm-up 标准化参考（供漂移分数自标准化对齐）
    warmup_cross_model = fit_warmup_cross_model(
        values[:train_end], base_ar, diag_lag, support
    )
    warm_mean = values[:train_end].mean(axis=0)
    warm_std = values[:train_end].std(axis=0)
    warm_std = np.where(warm_std < 1e-6, 1.0, warm_std)
    reference = values[train_end - window : train_end]
    rng = np.random.default_rng(20_000 + seed)
    high = train_end - window + 1
    if high <= lag:
        raise ValueError("Base training phase is too short for null calibration")
    starts = rng.integers(lag, high, size=null_windows)
    marginal_null = np.asarray(
        [marginal_score(reference, values[start : start + window]) for start in starts]
    )
    dependency_null = np.asarray(
        [
            dependency_drift_score(
                values[start : start + window],
                base_ar,
                diag_lag,
                support,
                warmup_cross_model,
                warm_mean,
                warm_std,
            )
            for start in starts
        ]
    )
    return {
        "reference": reference,
        "base_parameters": base_parameters,
        "base_ar": base_ar,
        "warmup_cross_model": warmup_cross_model,
        "warm_mean": warm_mean,
        "warm_std": warm_std,
        "support": support,
        "diag_lag": diag_lag,
        "marginal_null": marginal_null,
        "dependency_null": dependency_null,
        "threshold_marginal": float(np.quantile(marginal_null, 1.0 - alpha)),
        "threshold_dependency": float(np.quantile(dependency_null, 1.0 - alpha)),
    }


def diagnose_stage(
    current: np.ndarray,
    calibration: dict[str, object],
    alpha: float,
    uncertain_alpha: float,
    minimum_effect_z: float = 2.5,
    dependency_positivity_gate: bool = True,
) -> dict[str, object]:
    marginal = marginal_score(calibration["reference"], current)
    dependency = dependency_drift_score(
        current,
        calibration["base_ar"],
        calibration["diag_lag"],
        calibration["support"],
        calibration["warmup_cross_model"],
        calibration["warm_mean"],
        calibration["warm_std"],
    )
    marginal_null = calibration["marginal_null"]
    dependency_null = calibration["dependency_null"]
    p_marginal = float(
        (1 + np.sum(marginal_null >= marginal)) / (len(marginal_null) + 1)
    )
    p_dependency = float(
        (1 + np.sum(dependency_null >= dependency))
        / (len(dependency_null) + 1)
    )
    def robust_z(value: float, null: np.ndarray) -> float:
        center = float(np.median(null))
        mad = float(np.median(np.abs(null - center)))
        scale = max(1.4826 * mad, float(np.std(null, ddof=1)) * 0.25, 1e-8)
        return (value - center) / scale

    z_marginal = robust_z(marginal, marginal_null)
    z_dependency = robust_z(dependency, dependency_null)
    # A tail p-value alone is unstable when a null distribution is very narrow.
    # Requiring a standardized effect prevents systematic graph routing caused by
    # tiny but statistically extreme dependency-score fluctuations.
    marginal_flag = p_marginal <= alpha and z_marginal >= minimum_effect_z
    dependency_flag = p_dependency <= alpha and z_dependency >= minimum_effect_z
    # 正增益门控：只有当增量跨变量可预测性严格为正时才允许图更新，
    # 避免岭回归劣于自身基线时仍触发（审稿人指出的负分触发问题）。
    if dependency_positivity_gate and dependency <= 0.0:
        dependency_flag = False
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
    return {
        "routed_action": action,
        "score_marginal": marginal,
        "score_dependency": dependency,
        "z_marginal": z_marginal,
        "z_dependency": z_dependency,
        "p_marginal": p_marginal,
        "p_dependency": p_dependency,
        "uncertain": bool(uncertain),
        "dependency_positivity_gate": bool(dependency_positivity_gate),
        "threshold_marginal": calibration["threshold_marginal"],
        "threshold_dependency": calibration["threshold_dependency"],
        "minimum_effect_z": minimum_effect_z,
    }


def combine_datasets(left: TensorDataset, right: TensorDataset) -> TensorDataset:
    return TensorDataset(
        torch.cat((left.tensors[0], right.tensors[0]), dim=0),
        torch.cat((left.tensors[1], right.tensors[1]), dim=0),
    )


def sample_dataset(
    dataset: TensorDataset,
    size: int,
    rng: np.random.Generator,
) -> TensorDataset:
    count = len(dataset)
    if count <= size:
        return dataset
    indices = torch.from_numpy(rng.choice(count, size=size, replace=False))
    return TensorDataset(dataset.tensors[0][indices], dataset.tensors[1][indices])


def refresh_replay(
    replay: TensorDataset,
    current: TensorDataset,
    size: int,
    rng: np.random.Generator,
) -> TensorDataset:
    return sample_dataset(combine_datasets(replay, current), size, rng)


def adapt_copy(
    model: DecoupledSelectiveGraphForecaster,
    action: str,
    train_set: TensorDataset,
    validation_set: TensorDataset,
    device: torch.device,
    epochs: int,
    random_seed: int,
) -> tuple[DecoupledSelectiveGraphForecaster, int, float]:
    set_seed(random_seed)
    adapted = copy.deepcopy(model).to(device)
    trainable = configure_action(adapted, action)
    started = time.perf_counter()
    if trainable > 0:
        adapted = fit(
            adapted,
            train_set,
            validation_set,
            device,
            epochs,
            action_learning_rate(action),
        )
    elapsed = time.perf_counter() - started
    return adapted, trainable, elapsed


def estimate_diagonal_fisher(
    model: DecoupledSelectiveGraphForecaster,
    dataset: TensorDataset,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Estimate a diagonal empirical Fisher using only previously observed data."""
    model = model.to(device)
    configure_action(model, "full")
    fisher = {
        name: torch.zeros_like(parameter, device=device)
        for name, parameter in model.named_parameters()
    }
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    batches = 0
    model.train()
    for history, target in loader:
        history = history.to(device)
        target = target.to(device)
        model.zero_grad(set_to_none=True)
        loss = torch.mean((model(history) - target) ** 2)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                fisher[name] += parameter.grad.detach() ** 2
        batches += 1
    for name in fisher:
        fisher[name] /= max(batches, 1)
    model.zero_grad(set_to_none=True)
    return fisher


def adapt_regularized_copy(
    model: DecoupledSelectiveGraphForecaster,
    train_set: TensorDataset,
    validation_set: TensorDataset,
    device: torch.device,
    epochs: int,
    random_seed: int,
    method: str,
    regularization: float,
    historical_set: TensorDataset | None = None,
) -> tuple[DecoupledSelectiveGraphForecaster, int, float]:
    """Full-model adaptation with an EWC or LwF continual-learning penalty."""
    if method not in {"ewc", "lwf"}:
        raise ValueError(method)
    set_seed(random_seed)
    teacher = copy.deepcopy(model).to(device).eval()
    adapted = copy.deepcopy(model).to(device)
    trainable = configure_action(adapted, "full")
    started = time.perf_counter()
    anchors = {
        name: parameter.detach().clone()
        for name, parameter in teacher.named_parameters()
    }
    fisher = None
    if method == "ewc":
        if historical_set is None:
            raise ValueError("EWC requires a historical set")
        fisher = estimate_diagonal_fisher(teacher, historical_set, device)

    optimizer = torch.optim.Adam(
        [parameter for parameter in adapted.parameters() if parameter.requires_grad],
        lr=action_learning_rate("full"),
        weight_decay=1e-5,
    )
    loader = DataLoader(train_set, batch_size=256, shuffle=True)
    best_state = copy.deepcopy(adapted.state_dict())
    best_validation = float("inf")
    stale = 0
    for _ in range(epochs):
        adapted.train()
        for history, target in loader:
            history = history.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = adapted(history)
            loss = torch.mean((prediction - target) ** 2)
            if method == "ewc":
                penalty = sum(
                    torch.sum(fisher[name] * (parameter - anchors[name]) ** 2)
                    for name, parameter in adapted.named_parameters()
                )
                loss = loss + regularization * penalty
            else:
                with torch.no_grad():
                    teacher_prediction = teacher(history)
                loss = loss + regularization * torch.mean(
                    (prediction - teacher_prediction) ** 2
                )
            loss.backward()
            optimizer.step()
        validation_mse, _ = evaluate(adapted, validation_set, device)
        if validation_mse < best_validation - 1e-7:
            best_validation = validation_mse
            best_state = copy.deepcopy(adapted.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    adapted.load_state_dict(best_state)
    return adapted, trainable, time.perf_counter() - started


def guarded_candidates(diagnosis: dict[str, object], candidate_alpha: float) -> set[str]:
    routed = str(diagnosis["routed_action"])
    candidates = {"freeze", routed}
    if routed == "joint":
        candidates.update(("time", "graph"))
    elif routed == "time" and diagnosis["p_dependency"] <= candidate_alpha:
        candidates.add("joint")
    elif routed == "graph" and diagnosis["p_marginal"] <= candidate_alpha:
        candidates.add("joint")
    elif routed == "freeze":
        if diagnosis["p_marginal"] <= candidate_alpha:
            candidates.add("time")
        if diagnosis["p_dependency"] <= candidate_alpha:
            candidates.add("graph")
    return candidates


def summarize_results(stage_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for (dataset, seed, method), group in stage_results.groupby(
        ["dataset", "seed", "method"], sort=False
    ):
        weights = group.test_elements.to_numpy(dtype=float)
        seed_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "mse": float(np.average(group.current_mse, weights=weights)),
                "mae": float(np.average(group.current_mae, weights=weights)),
                "mean_forgetting": float(group.forgetting.mean()),
                "final_forgetting": float(group.iloc[-1].forgetting),
                "adaptation_seconds": float(group.adaptation_seconds.sum()),
                "mean_trainable_fraction": float(group.trainable_fraction.mean()),
            }
        )
    seed_summary = pd.DataFrame(seed_rows)
    aggregate_rows = []
    metrics = (
        "mse",
        "mae",
        "mean_forgetting",
        "final_forgetting",
        "adaptation_seconds",
        "mean_trainable_fraction",
    )
    for (dataset, method), group in seed_summary.groupby(["dataset", "method"], sort=False):
        row = {"dataset": dataset, "method": method, "seeds": int(len(group))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        aggregate_rows.append(row)
    method_summary = pd.DataFrame(aggregate_rows).sort_values(["dataset", "mse_mean"])
    return seed_summary, method_summary


def run_seed(
    values: np.ndarray,
    dataset_name: str,
    warmup_end: int,
    seed: int,
    device: torch.device,
    stages: int,
    lag: int,
    base_epochs: int,
    adapt_epochs: int,
    adaptation_points: int,
    validation_points: int,
    null_windows: int,
    alpha: float,
    uncertain_alpha: float,
    candidate_alpha: float,
    replay_size: int,
    graph_top_k: int,
    ewc_lambda: float,
    lwf_lambda: float,
    diag_lag: int,
    dependency_positivity_gate: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base_train_end = int(warmup_end * 0.70)
    base_validation_end = int(warmup_end * 0.85)
    base_train = make_windows(values, 0, base_train_end, lag)
    base_validation = make_windows(values, base_train_end, base_validation_end, lag)
    base_test = make_windows(values, base_validation_end, warmup_end, lag)
    support = infer_support(values[:base_train_end], graph_top_k)
    calibration = calibrate_natural_null(
        values,
        base_train_end,
        lag,
        adaptation_points,
        null_windows,
        alpha,
        seed,
        support,
        diag_lag,
    )

    set_seed(seed)
    base_model = DecoupledSelectiveGraphForecaster(
        values.shape[1], lag, support
    ).to(device)
    configure_action(base_model, "base")
    base_model = fit(base_model, base_train, base_validation, device, base_epochs, 2e-3)
    base_mse, base_mae = evaluate(base_model, base_test, device)
    full_parameters = sum(parameter.numel() for parameter in base_model.parameters())
    method_models = {method: copy.deepcopy(base_model).to(device) for method in METHODS}
    rng = np.random.default_rng(30_000 + seed)
    replay = sample_dataset(base_train, replay_size, rng)

    stage_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    stage_edges = np.linspace(warmup_end, len(values), stages + 1, dtype=int)
    for stage in range(stages):
        stage_start = int(stage_edges[stage])
        stage_end = int(stage_edges[stage + 1])
        adaptation_end = stage_start + adaptation_points
        validation_end = adaptation_end + validation_points
        if validation_end + lag >= stage_end:
            raise ValueError(
                f"Stage {stage} is too short for adaptation/validation/test split"
            )
        adaptation_set = make_windows(values, stage_start, adaptation_end, lag)
        validation_set = make_windows(values, adaptation_end, validation_end, lag)
        current_test = make_windows(values, validation_end, stage_end, lag)
        diagnosis = diagnose_stage(
            values[stage_start:adaptation_end],
            calibration,
            alpha,
            uncertain_alpha,
            dependency_positivity_gate=dependency_positivity_gate,
        )
        routing_rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "stage": stage,
                "stage_start": stage_start,
                "stage_end": stage_end,
                **diagnosis,
            }
        )

        actions = {f"always_{action}": action for action in FIXED_ACTIONS}
        actions["sca_router"] = str(diagnosis["routed_action"])
        for method, action in actions.items():
            adapted, trainable, elapsed = adapt_copy(
                method_models[method],
                action,
                adaptation_set,
                validation_set,
                device,
                adapt_epochs,
                seed * 10_000 + stage * 100 + METHODS.index(method),
            )
            method_models[method] = adapted
            stage_rows.append(
                evaluate_stage(
                    adapted,
                    dataset_name,
                    seed,
                    stage,
                    method,
                    action,
                    current_test,
                    base_test,
                    base_mse,
                    base_mae,
                    trainable,
                    full_parameters,
                    elapsed,
                    device,
                    "",
                )
            )

        replay_train = combine_datasets(adaptation_set, replay)
        replay_model, replay_trainable, replay_elapsed = adapt_copy(
            method_models["experience_replay"],
            "full",
            replay_train,
            validation_set,
            device,
            adapt_epochs,
            seed * 10_000 + stage * 100 + METHODS.index("experience_replay"),
        )
        method_models["experience_replay"] = replay_model
        stage_rows.append(
            evaluate_stage(
                replay_model,
                dataset_name,
                seed,
                stage,
                "experience_replay",
                "full+replay",
                current_test,
                base_test,
                base_mse,
                base_mae,
                replay_trainable,
                full_parameters,
                replay_elapsed,
                device,
                f"buffer={len(replay)}",
            )
        )

        for method, regularization in (("ewc", ewc_lambda), ("lwf", lwf_lambda)):
            regularized_model, regularized_trainable, regularized_elapsed = (
                adapt_regularized_copy(
                    method_models[method],
                    adaptation_set,
                    validation_set,
                    device,
                    adapt_epochs,
                    seed * 10_000 + stage * 100 + METHODS.index(method),
                    method,
                    regularization,
                    replay if method == "ewc" else None,
                )
            )
            method_models[method] = regularized_model
            stage_rows.append(
                evaluate_stage(
                    regularized_model,
                    dataset_name,
                    seed,
                    stage,
                    method,
                    f"full+{method}",
                    current_test,
                    base_test,
                    base_mse,
                    base_mae,
                    regularized_trainable,
                    full_parameters,
                    regularized_elapsed,
                    device,
                    f"lambda={regularization:g}",
                )
            )

        candidate_models = []
        candidate_total_time = 0.0
        candidates = guarded_candidates(diagnosis, candidate_alpha)
        for candidate_index, action in enumerate(sorted(candidates)):
            adapted, trainable, elapsed = adapt_copy(
                method_models["guarded_router"],
                action,
                adaptation_set,
                validation_set,
                device,
                adapt_epochs,
                seed * 10_000 + stage * 100 + 50 + candidate_index,
            )
            validation_mse, _ = evaluate(adapted, validation_set, device)
            score = validation_mse + 0.002 * trainable / full_parameters
            candidate_models.append((score, action, adapted, trainable))
            candidate_total_time += elapsed
        _, chosen_action, guarded_model, guarded_trainable = min(
            candidate_models, key=lambda item: item[0]
        )
        method_models["guarded_router"] = guarded_model
        stage_rows.append(
            evaluate_stage(
                guarded_model,
                dataset_name,
                seed,
                stage,
                "guarded_router",
                chosen_action,
                current_test,
                base_test,
                base_mse,
                base_mae,
                guarded_trainable,
                full_parameters,
                candidate_total_time,
                device,
                ",".join(sorted(candidates)),
            )
        )
        replay = refresh_replay(replay, adaptation_set, replay_size, rng)
    return stage_rows, routing_rows


def evaluate_stage(
    model: DecoupledSelectiveGraphForecaster,
    dataset_name: str,
    seed: int,
    stage: int,
    method: str,
    action: str,
    current_test: TensorDataset,
    base_test: TensorDataset,
    base_mse: float,
    base_mae: float,
    trainable: int,
    full_parameters: int,
    elapsed: float,
    device: torch.device,
    detail: str,
) -> dict[str, object]:
    current_mse, current_mae = evaluate(model, current_test, device)
    history_mse, history_mae = evaluate(model, base_test, device)
    return {
        "dataset": dataset_name,
        "seed": seed,
        "stage": stage,
        "method": method,
        "action": action,
        "current_mse": current_mse,
        "current_mae": current_mae,
        "history_mse": history_mse,
        "history_mae": history_mae,
        "base_mse": base_mse,
        "base_mae": base_mae,
        "forgetting": max(history_mse - base_mse, 0.0),
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / full_parameters,
        "adaptation_seconds": elapsed,
        "test_elements": int(len(current_test) * model.n_nodes),
        "detail": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.25)
    parser.add_argument("--lag", type=int, default=60)
    parser.add_argument("--base-epochs", type=int, default=25)
    parser.add_argument("--adapt-epochs", type=int, default=8)
    parser.add_argument("--adaptation-points", type=int, default=512)
    parser.add_argument("--validation-points", type=int, default=512)
    parser.add_argument("--null-windows", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--uncertain-alpha", type=float, default=0.10)
    parser.add_argument("--candidate-alpha", type=float, default=0.25)
    parser.add_argument("--replay-size", type=int, default=512)
    parser.add_argument("--graph-top-k", type=int, default=4)
    parser.add_argument("--ewc-lambda", type=float, default=10.0)
    parser.add_argument("--lwf-lambda", type=float, default=1.0)
    parser.add_argument(
        "--diag-lag",
        type=int,
        default=None,
        help="诊断层滞后阶数；默认 min(lag, 12)，与论文 controlled stream 的 L=12 对齐",
    )
    parser.add_argument(
        "--dependency-positivity-gate",
        action="store_true",
        default=True,
        help="要求增量跨变量可预测性严格为正才允许图更新",
    )
    parser.add_argument(
        "--no-dependency-positivity-gate",
        dest="dependency_positivity_gate",
        action="store_false",
        help="关闭正增益门控（用于对照）",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or args.data.stem
    values, data_metadata = load_standardized_csv(args.data, args.warmup_ratio)
    device = torch.device(args.device)
    diag_lag = args.diag_lag if args.diag_lag is not None else min(args.lag, 12)

    stage_rows = []
    routing_rows = []
    for seed in range(args.seeds):
        print(f"Running natural stream {dataset_name} seed {seed} on {device}", flush=True)
        seed_stages, seed_routing = run_seed(
            values,
            dataset_name,
            data_metadata["warmup_end"],
            seed,
            device,
            args.stages,
            args.lag,
            args.base_epochs,
            args.adapt_epochs,
            args.adaptation_points,
            args.validation_points,
            args.null_windows,
            args.alpha,
            args.uncertain_alpha,
            args.candidate_alpha,
            args.replay_size,
            args.graph_top_k,
            args.ewc_lambda,
            args.lwf_lambda,
            diag_lag,
            args.dependency_positivity_gate,
        )
        stage_rows.extend(seed_stages)
        routing_rows.extend(seed_routing)

    stage_results = pd.DataFrame(stage_rows)
    routing_results = pd.DataFrame(routing_rows)
    seed_summary, method_summary = summarize_results(stage_results)
    stage_results.to_csv(args.output / "stage_results.csv", index=False)
    routing_results.to_csv(args.output / "routing_decisions.csv", index=False)
    seed_summary.to_csv(args.output / "seed_summary.csv", index=False)
    method_summary.to_csv(args.output / "method_summary.csv", index=False)
    metadata = {
        "evidence_type": "newly_run",
        "dataset": dataset_name,
        "source": str(args.data.resolve()),
        "data": data_metadata,
        "protocol": {
            "warmup_ratio": args.warmup_ratio,
            "warmup_split": "70% base train / 15% validation / 15% historical test",
            "stream_stages": args.stages,
            "stage_split": {
                "adaptation_points": args.adaptation_points,
                "validation_points": args.validation_points,
                "test": "remaining points in each chronological stage",
            },
            "lag": args.lag,
            "horizon": 1,
            "diagnostic": {
                "diag_lag": diag_lag,
                "dependency_positivity_gate": bool(args.dependency_positivity_gate),
                "score": "cross-variable predictive-structure drift vs warm-up snapshot (block self-normalized to remove marginal shift)",
            },
            "replay_size": args.replay_size,
            "ewc_lambda": args.ewc_lambda,
            "lwf_lambda": args.lwf_lambda,
            "normalization": data_metadata["normalization"],
            "candidate_selection": "stage validation MSE + 0.002 * trainable fraction",
        },
        "training": {
            "seeds": args.seeds,
            "base_epochs": args.base_epochs,
            "adapt_epochs": args.adapt_epochs,
            "device": str(device),
            "torch": torch.__version__,
        },
        "comparability_warning": (
            "Stage-wise natural-stream protocol; do not directly rank against paper-reported "
            "per-round online results without a protocol-matched rerun."
        ),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(method_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
