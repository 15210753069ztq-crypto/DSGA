from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from e0_oracle.run_e0 import pairs, ridge_estimate


SCENARIOS = ("stable", "marginal", "dependency", "joint")
ACTIONS = ("freeze", "time", "graph", "joint", "full")
EXPECTED_ACTION = {
    "stable": "freeze",
    "marginal": "time",
    "dependency": "graph",
    "joint": "joint",
}
ACTION_CLASS = {"freeze": 0, "time": 1, "graph": 2, "joint": 3}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_windows(values: np.ndarray, start: int, end: int, lag: int) -> TensorDataset:
    histories = []
    targets = []
    for index in range(max(start, lag), end):
        histories.append(values[index - lag : index].T)
        targets.append(values[index])
    return TensorDataset(
        torch.from_numpy(np.asarray(histories, dtype=np.float32)),
        torch.from_numpy(np.asarray(targets, dtype=np.float32)),
    )


class SelectiveGraphForecaster(nn.Module):
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
        logits = torch.where(prior > 0, torch.full_like(prior, 2.2), torch.full_like(prior, -2.2))
        logits.fill_diagonal_(-8.0)
        self.base_graph_logits = nn.Parameter(logits)
        self.register_buffer("off_diagonal", 1.0 - torch.eye(n_nodes))
        self.readout = nn.Linear(2 * hidden, 1)

        self.time_scale = nn.Parameter(torch.zeros(n_nodes))
        self.time_bias = nn.Parameter(torch.zeros(n_nodes))
        generator = torch.Generator().manual_seed(12345)
        # Keep the initial graph unchanged while giving the low-rank adapter a
        # usable first-step gradient through graph_v.
        self.graph_u = nn.Parameter(torch.randn(n_nodes, graph_rank, generator=generator) * 0.5)
        self.graph_v = nn.Parameter(torch.zeros(n_nodes, graph_rank))

    def adjacency(self) -> torch.Tensor:
        delta = 5.0 * (self.graph_u @ self.graph_v.T)
        adjacency = torch.sigmoid(self.base_graph_logits + delta) * self.off_diagonal
        return adjacency / (adjacency.sum(dim=1, keepdim=True) + 1e-6)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        hidden = self.temporal(history)
        messages = torch.einsum("ij,bjh->bih", self.adjacency(), hidden)
        prediction = self.readout(torch.cat((hidden, messages), dim=-1)).squeeze(-1)
        return prediction * (1.0 + self.time_scale) + self.time_bias


def configure_action(model: SelectiveGraphForecaster, action: str) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if action == "base":
        for parameter in model.temporal.parameters():
            parameter.requires_grad = True
        model.base_graph_logits.requires_grad = True
        for parameter in model.readout.parameters():
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
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: TensorDataset,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    squared = 0.0
    absolute = 0.0
    count = 0
    for history, target in loader:
        history = history.to(device)
        target = target.to(device)
        prediction = model(history)
        squared += torch.sum((prediction - target) ** 2).item()
        absolute += torch.sum(torch.abs(prediction - target)).item()
        count += target.numel()
    return squared / count, absolute / count


def fit(
    model: SelectiveGraphForecaster,
    train_set: TensorDataset,
    validation_set: TensorDataset,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> SelectiveGraphForecaster:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return model
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)
    loader = DataLoader(train_set, batch_size=256, shuffle=True)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    patience = 5
    stale = 0
    for _ in range(epochs):
        model.train()
        for history, target in loader:
            history = history.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history)
            loss = torch.mean((prediction - target) ** 2)
            loss = loss + 1e-4 * (
                torch.mean(model.graph_u**2) + torch.mean(model.graph_v**2)
            )
            loss.backward()
            optimizer.step()
        validation, _ = evaluate(model, validation_set, device)
        if validation < best_validation - 1e-7:
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model


def marginal_score(reference: np.ndarray, current: np.ndarray) -> float:
    mean_shift = np.linalg.norm(current.mean(axis=0) - reference.mean(axis=0)) / np.sqrt(
        reference.shape[1]
    )
    reference_covariance = np.cov(reference, rowvar=False)
    current_covariance = np.cov(current, rowvar=False)
    covariance_shift = np.linalg.norm(
        current_covariance - reference_covariance,
        ord="fro",
    ) / max(np.linalg.norm(reference_covariance, ord="fro"), 1e-12)
    return float(mean_shift + covariance_shift)


def dependency_score(current: np.ndarray, base_parameters: np.ndarray) -> float:
    x_augmented, y = pairs(current)
    residual = y - base_parameters @ x_augmented
    midpoint = residual.shape[1] // 2
    x_train = x_augmented[:-1, :midpoint]
    x_test = x_augmented[:-1, midpoint:]
    r_train = residual[:, :midpoint]
    r_test = residual[:, midpoint:]
    baseline = r_train.mean(axis=1, keepdims=True)
    baseline_mse = float(np.mean((r_test - baseline) ** 2))
    train_augmented = np.vstack((x_train, np.ones(x_train.shape[1])))
    test_augmented = np.vstack((x_test, np.ones(x_test.shape[1])))
    residual_model = ridge_estimate(train_augmented, r_train, ridge=10.0)
    model_mse = float(np.mean((r_test - residual_model @ test_augmented) ** 2))
    return (baseline_mse - model_mse) / max(baseline_mse, 1e-12)


def _stack_lag_features(values: np.ndarray, lag: int) -> np.ndarray:
    """返回形状 (T-lag, N, lag) 的滞后特征矩阵。

    第 t 行（t 从 0 起）对应预测时刻 t+lag，包含 values[t : t+lag] 的 lag 个历史点，
    即 features[t, :, k] = values[t + lag - 1 - k]，与 y = values[lag:] 对齐。
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if values.shape[0] <= lag:
        raise ValueError("window is shorter than lag")
    return np.stack(
        [values[lag - 1 - k : values.shape[0] - 1 - k] for k in range(lag)],
        axis=2,
    )


def fit_temporal_matched_ar(
    base_values: np.ndarray, diag_lag: int, ridge: float = 1.0
) -> np.ndarray:
    """在 warm-up 数据上拟合每变量独立的多阶自回归系数（含截距）。

    返回 shape (N, diag_lag + 1) 的矩阵，最后一列为截距。
    与裸 VAR(1) 不同，这里给每个变量提供 diag_lag 阶自身历史，
    从而提供与 temporal 适配器同阶的时间表达能力，消除线性时间欠拟合混淆。
    """
    n_vars = base_values.shape[1]
    x_lag = _stack_lag_features(base_values, diag_lag)  # (T-lag, N, lag)
    y = base_values[diag_lag:]  # (T-lag, N)
    coefficients = np.empty((n_vars, diag_lag + 1), dtype=np.float64)
    for i in range(n_vars):
        design = np.hstack(
            (x_lag[:, i, :], np.ones((x_lag.shape[0], 1)))
        )  # (T-lag, lag+1)
        coefficients[i] = _ridge_solve(design, y[:, i], ridge)
    return coefficients


def _ridge_solve(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(x.shape[1])
    penalty[-1, -1] = ridge * 0.1  # 不惩罚截距
    gram = x.T @ x + penalty
    return np.linalg.solve(gram, x.T @ y)


def temporal_matched_residuals(
    current: np.ndarray, base_ar: np.ndarray, diag_lag: int
) -> np.ndarray:
    """用冻结的每变量 AR 基线对当前窗口生成残差，返回 shape (T-diag_lag, N)。"""
    n_vars = current.shape[1]
    x_lag = _stack_lag_features(current, diag_lag)
    prediction = np.empty((x_lag.shape[0], n_vars), dtype=np.float64)
    for i in range(n_vars):
        design = np.hstack((x_lag[:, i, :], np.ones((x_lag.shape[0], 1))))
        prediction[:, i] = design @ base_ar[i]
    return current[diag_lag:] - prediction


def dependency_score_cross(
    current: np.ndarray,
    base_ar: np.ndarray,
    diag_lag: int,
    support: np.ndarray,
    ridge: float = 10.0,
) -> float:
    """增量跨变量残差可预测性分数。

    对每个变量 i，在 chronological split 上比较：
      - self 模型：只用变量 i 自身 lag-1..lag-diag_lag 预测残差；
      - full 模型：在 self 特征上加入支撑集邻居的滞后。
    仅当 full 相对 self 有样本外增益时分数为正，从而把"自身自回归动态变化"
    排除在依赖证据之外。返回各变量相对增益的均值（裁剪到 [-1, 1]）。
    """
    residuals = temporal_matched_residuals(current, base_ar, diag_lag)
    n_samples, n_vars = residuals.shape
    if n_samples < 4 or n_vars < 2:
        return 0.0
    x_lag = _stack_lag_features(current, diag_lag)  # (T-diag_lag, N, lag)
    # residuals 与 x_lag 的时间索引对齐（都是 lag 之后开始的 T-diag_lag 行）
    midpoint = n_samples // 2
    if midpoint < 2:
        return 0.0
    gains = []
    for i in range(n_vars):
        self_feat = x_lag[:, i, :]  # (n_samples, lag)
        neighbors = np.flatnonzero(support[i])
        neighbors = neighbors[neighbors != i]
        if neighbors.size == 0:
            continue
        cross_feat = x_lag[:, neighbors, :].reshape(n_samples, -1)
        design_self = np.hstack((self_feat, np.ones((n_samples, 1))))
        design_full = np.hstack((self_feat, cross_feat, np.ones((n_samples, 1))))
        y = residuals[:, i]
        mse_self = _chrono_mse(design_self, y, midpoint, ridge)
        mse_full = _chrono_mse(design_full, y, midpoint, ridge)
        if mse_self <= 1e-12:
            continue
        gain = (mse_self - mse_full) / mse_self
        gains.append(float(np.clip(gain, -1.0, 1.0)))
    if not gains:
        return 0.0
    return float(np.mean(gains))


def _chrono_mse(
    design: np.ndarray, y: np.ndarray, midpoint: int, ridge: float
) -> float:
    """在前半段拟合、后半段评估，避免把模型容量变成表面上的依赖证据。"""
    x_train, x_test = design[:midpoint], design[midpoint:]
    y_train, y_test = y[:midpoint], y[midpoint:]
    if x_train.shape[0] < 2:
        return float(np.inf)
    try:
        weights = _ridge_solve(x_train, y_train, ridge)
    except np.linalg.LinAlgError:
        return float(np.inf)
    return float(np.mean((y_test - x_test @ weights) ** 2))


def fit_warmup_cross_model(
    warmup: np.ndarray,
    base_ar: np.ndarray,
    diag_lag: int,
    support: np.ndarray,
    ridge: float = 50.0,
) -> dict[int, np.ndarray | None]:
    """在 warm-up 上拟合并冻结"每变量跨变量残差模型"。

    对每个变量 i，用其支撑邻居的 lag-1 预测该变量的 temporal-matched 残差，
    得到一组冻结权重，作为"历史跨变量预测结构"的快照。后续诊断比较当前块
    对该快照的失配程度，从而检测结构是否"变化"（而非是否"存在"）。
    """
    residuals = temporal_matched_residuals(warmup, base_ar, diag_lag)
    n_samples = residuals.shape[0]
    x_lag = _stack_lag_features(warmup, diag_lag)
    models: dict[int, np.ndarray | None] = {}
    for i in range(warmup.shape[1]):
        neighbors = np.flatnonzero(support[i])
        neighbors = neighbors[neighbors != i]
        if neighbors.size == 0:
            models[i] = None
            continue
        features = x_lag[:, neighbors, 0].reshape(n_samples, -1)  # 邻居 lag-1
        design = np.hstack((features, np.ones((n_samples, 1))))
        try:
            models[i] = _ridge_solve(design, residuals[:, i], ridge)
        except np.linalg.LinAlgError:
            models[i] = None
    return models


def dependency_drift_score(
    current: np.ndarray,
    base_ar: np.ndarray,
    diag_lag: int,
    support: np.ndarray,
    warmup_cross_model: dict[int, np.ndarray | None],
    warm_mean: np.ndarray,
    warm_std: np.ndarray,
    ridge: float = 50.0,
) -> float:
    """跨变量预测结构漂移分数（相对 warm-up 快照）。

    与 ``dependency_score_cross`` 检测"是否存在跨变量结构"不同，本分数检测
    "跨变量预测结构相对 warm-up 是否发生变化"，这才是依赖漂移的本质。

    步骤：
      1. 块内自标准化并对齐回 warm-up 尺度，剥离纯 marginal（均值/尺度）漂移；
      2. 用冻结的 warm-up 跨变量模型在当前块后半段预测残差，得到失配误差；
      3. 在当前块前半段重新拟合跨变量模型，在后半段评估；
      4. 分数 = (失配误差 - 重拟合误差) / 失配误差，各变量取均值并裁剪到 [-1, 1]。

    结构未变时冻结模型仍好用，重拟合挽回有限，分数≈0 或负；结构变化时冻结模型
    失配、重拟合显著挽回，分数为正。
    """
    block_mean = current.mean(axis=0)
    block_std = current.std(axis=0)
    block_std = np.where(block_std < 1e-6, 1.0, block_std)
    aligned = (current - block_mean) / block_std * warm_std + warm_mean

    residuals = temporal_matched_residuals(aligned, base_ar, diag_lag)
    n_samples, n_vars = residuals.shape
    if n_samples < 4 or n_vars < 2:
        return 0.0
    x_lag = _stack_lag_features(aligned, diag_lag)
    midpoint = n_samples // 2
    if midpoint < 2:
        return 0.0
    gains = []
    for i in range(n_vars):
        frozen = warmup_cross_model.get(i)
        if frozen is None:
            continue
        neighbors = np.flatnonzero(support[i])
        neighbors = neighbors[neighbors != i]
        if neighbors.size == 0:
            continue
        features = x_lag[:, neighbors, 0].reshape(n_samples, -1)
        design = np.hstack((features, np.ones((n_samples, 1))))
        y = residuals[:, i]
        mse_frozen = float(np.mean((y[midpoint:] - design[midpoint:] @ frozen) ** 2))
        try:
            refit = _ridge_solve(design[:midpoint], y[:midpoint], ridge)
        except np.linalg.LinAlgError:
            continue
        mse_refit = float(np.mean((y[midpoint:] - design[midpoint:] @ refit) ** 2))
        if mse_frozen <= 1e-9:
            continue
        gains.append(float(np.clip((mse_frozen - mse_refit) / mse_frozen, -1.0, 1.0)))
    if not gains:
        return 0.0
    return float(np.mean(gains))


def route(
    base_values: np.ndarray,
    current: np.ndarray,
    thresholds: tuple[float, float],
) -> tuple[str, float, float]:
    base_x, base_y = pairs(base_values[:4000])
    base_parameters = ridge_estimate(base_x, base_y, ridge=1.0)
    reference = base_values[5000:5512]
    marginal = marginal_score(reference, current[:512])
    dependency = dependency_score(current[:512], base_parameters)
    marginal_flag = marginal > thresholds[0]
    dependency_flag = dependency > thresholds[1]
    if marginal_flag and dependency_flag:
        action = "joint"
    elif marginal_flag:
        action = "time"
    elif dependency_flag:
        action = "graph"
    else:
        action = "freeze"
    return action, marginal, dependency


def run_seed(
    dataset_path: Path,
    seed: int,
    thresholds: tuple[float, float],
    device: torch.device,
    base_epochs: int,
    adapt_epochs: int,
    adaptation_points: int,
    lag: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    payload = np.load(dataset_path)
    base_values = payload["base"]
    support = payload["base_support"].astype(bool)
    base_train = make_windows(base_values, 0, 4000, lag)
    base_validation = make_windows(base_values, 4000, 5000, lag)
    base_test = make_windows(base_values, 5000, 6000, lag)

    set_seed(seed)
    model = SelectiveGraphForecaster(base_values.shape[1], lag, support).to(device)
    configure_action(model, "base")
    model = fit(model, base_train, base_validation, device, base_epochs, 2e-3)
    base_mse, base_mae = evaluate(model, base_test, device)
    full_parameters = sum(parameter.numel() for parameter in model.parameters())

    action_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        values = payload[scenario]
        adaptation_set = make_windows(values, 0, adaptation_points, lag)
        validation_set = make_windows(values, adaptation_points, 1024, lag)
        current_test = make_windows(values, 2000, 6000, lag)
        routed_action, score_marginal, score_dependency = route(
            base_values,
            values,
            thresholds,
        )
        routing_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "routed_action": routed_action,
                "expected_action": EXPECTED_ACTION[scenario],
                "score_marginal": score_marginal,
                "score_dependency": score_dependency,
            }
        )
        for action in ACTIONS:
            set_seed(seed * 100 + ACTIONS.index(action))
            adapted = copy.deepcopy(model).to(device)
            trainable = configure_action(adapted, action)
            started = time.perf_counter()
            if action != "freeze":
                adapted = fit(
                    adapted,
                    adaptation_set,
                    validation_set,
                    device,
                    adapt_epochs,
                    (
                        1e-2
                        if action == "graph"
                        else 6e-3
                        if action == "joint"
                        else 3e-3
                        if action == "time"
                        else 8e-4
                    ),
                )
            elapsed = time.perf_counter() - started
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


def build_methods(actions: pd.DataFrame, routing: pd.DataFrame) -> pd.DataFrame:
    merged = actions.merge(routing, on=["seed", "scenario", "expected_action"])
    rows = []
    for action in ACTIONS:
        fixed = merged[merged.action == action].copy()
        fixed["method"] = f"always_{action}"
        rows.append(fixed)
    router = merged[merged.action == merged.routed_action].copy()
    router["method"] = "sca_router"
    rows.append(router)
    oracle = merged[merged.action == merged.expected_action].copy()
    oracle["method"] = "mechanism_oracle"
    rows.append(oracle)
    keys = ["seed", "scenario"]
    best = merged.loc[merged.groupby(keys).utility.idxmin()].copy()
    best["method"] = "best_action_oracle"
    rows.append(best)
    return pd.concat(rows, ignore_index=True)


def summarize(methods: pd.DataFrame, routing: pd.DataFrame) -> dict[str, object]:
    overall = (
        methods.groupby("method")
        .agg(
            current_mse=("current_mse", "mean"),
            current_mae=("current_mae", "mean"),
            forgetting=("forgetting", "mean"),
            utility=("utility", "mean"),
            adaptation_seconds=("adaptation_seconds", "mean"),
        )
        .reset_index()
        .sort_values("utility")
    )
    fixed = overall[overall.method.str.startswith("always_")]
    best_fixed = str(fixed.iloc[0].method)
    router_utility = float(overall.loc[overall.method == "sca_router", "utility"].iloc[0])
    best_fixed_utility = float(overall.loc[overall.method == best_fixed, "utility"].iloc[0])
    oracle_utility = float(
        overall.loc[overall.method == "mechanism_oracle", "utility"].iloc[0]
    )
    routing_accuracy = float((routing.routed_action == routing.expected_action).mean())
    routing_f1 = float(
        f1_score(
            routing.expected_action.map(ACTION_CLASS),
            routing.routed_action.map(ACTION_CLASS),
            average="macro",
        )
    )
    if router_utility < best_fixed_utility and routing_accuracy >= 0.60:
        support = "trend_support"
    else:
        support = "not_supported"
    return {
        "support_level": support,
        "routing_accuracy": routing_accuracy,
        "routing_macro_f1": routing_f1,
        "best_fixed_method": best_fixed,
        "router_minus_best_fixed_utility": router_utility - best_fixed_utility,
        "router_minus_mechanism_oracle_utility": router_utility - oracle_utility,
        "overall": overall.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--threshold-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-epochs", type=int, default=35)
    parser.add_argument("--adapt-epochs", type=int, default=15)
    parser.add_argument("--adaptation-points", type=int, default=512)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    threshold_frame = pd.read_csv(args.threshold_summary)
    threshold_row = threshold_frame[threshold_frame.window == 512].iloc[0]
    thresholds = (
        float(threshold_row.threshold_marginal),
        float(threshold_row.threshold_dependency),
    )
    device = torch.device(args.device)
    action_rows = []
    routing_rows = []
    for seed in range(args.seeds):
        print(f"Running neural GNN seed {seed} on {device}", flush=True)
        seed_actions, seed_routing = run_seed(
            args.data / f"seed_{seed:02d}.npz",
            seed,
            thresholds,
            device,
            args.base_epochs,
            args.adapt_epochs,
            args.adaptation_points,
            args.lag,
        )
        action_rows.extend(seed_actions)
        routing_rows.extend(seed_routing)
    actions = pd.DataFrame(action_rows)
    routing = pd.DataFrame(routing_rows)
    methods = build_methods(actions, routing)
    assessment = summarize(methods, routing)
    actions.to_csv(args.output / "neural_action_results.csv", index=False)
    routing.to_csv(args.output / "neural_routing_decisions.csv", index=False)
    methods.to_csv(args.output / "neural_method_results.csv", index=False)
    (args.output / "neural_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    print(json.dumps(assessment, indent=2), flush=True)


if __name__ == "__main__":
    main()
