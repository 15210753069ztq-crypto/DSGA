from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_natural_stream_benchmark import (
    calibrate_natural_null,
    combine_datasets,
    diagnose_stage,
    infer_support,
    sample_dataset,
)
from run_neural_gnn_prediction import set_seed
from run_neural_gnn_prediction_v2 import action_learning_rate as _action_learning_rate_v2
from run_neural_gnn_prediction_v2 import configure_action as _configure_action_v2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dataset" / "PEMS" / "PEMS08.npz"
DEFAULT_OUTPUT = ROOT / "outputs" / "pems08_prequential_multihorizon"
HORIZON = 12
REPORT_STEPS = (3, 6, 12)
FORECAST_CONTROLS = ("DLinear-Online", "STGCN-Online", "PatchTST-Online")
GENERIC_CL = ("Online-Full", "ER", "EWC", "DER++")
SELECTIVE_METHODS = (
    "Always-Time",
    "Always-Graph",
    "Always-Joint",
    "DSGA-NoMemory",
    "DSGA",
)
METHODS = ("Frozen",) + FORECAST_CONTROLS + GENERIC_CL + SELECTIVE_METHODS
ACTION_FOR_FIXED = {
    "Always-Time": "time",
    "Always-Graph": "graph",
    "Always-Joint": "joint",
}
TIME_NAMES = ("time_scale", "time_bias", "mod_a1", "mod_b1", "mod_a2", "mod_b2")
GRAPH_NAMES = ("graph_u", "graph_v", "msg_down", "msg_up")
ADAPTER_NAMES = TIME_NAMES + GRAPH_NAMES
# Exploration switches (all off by default, which reproduces version v1).
MODEL_OPTIONS: dict[str, bool] = {
    "compact_time": False,
    "revin": False,
    "weight_mod": False,
    "msg_adapter": False,
}
FIT_OPTIONS: dict[str, object] = {"closed_form_time": False, "ridge": 1.0}
GATE_OPTIONS: dict[str, object] = {"enabled": False, "eta": 1.0}
DV_OPTIONS: dict[str, object] = {"mode": "off", "delta": 0.03, "eta": 1.0}
TUNE_OPTIONS: dict[str, float] = {"lr_scale": 1.0, "baseline_lr_scale": 1.0, "mod_rank": 2}


def action_learning_rate(action: str) -> float:
    return _action_learning_rate_v2(action) * float(TUNE_OPTIONS["lr_scale"])
UPGRADE_OPTIONS: dict[str, object] = {"tau": None}
# 显存不足时把每个 128 的训练批拆成若干小块累积梯度；数学上等价于整批更新（0 表示不拆）。
# eval_batch 只影响无梯度推理的分批大小，不改变任何结果。
MEMORY_OPTIONS: dict[str, int] = {"micro_batch": 0, "eval_batch": 256}


def configure_action(model: nn.Module, action: str) -> int:
    """v1 action gating plus the optional adapter modules."""
    _configure_action_v2(model, action)
    if action in ("time", "joint"):
        for name in TIME_NAMES:
            if hasattr(model, name):
                getattr(model, name).requires_grad = True
    if action in ("graph", "joint"):
        for name in GRAPH_NAMES:
            if hasattr(model, name):
                getattr(model, name).requires_grad = True
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_windows(
    values: np.ndarray,
    start: int,
    end: int,
    lag: int,
    horizon: int | None = None,
) -> TensorDataset:
    """Create windows whose complete target horizon lies inside [start, end)."""
    horizon = HORIZON if horizon is None else int(horizon)
    first = max(start, lag)
    last = end - horizon + 1
    if last <= first:
        raise ValueError(f"Window [{start}, {end}) is too short for L={lag}, H={horizon}")
    indices = np.arange(first, last, dtype=np.int64)
    histories = np.stack([values[index - lag : index].T for index in indices])
    targets = np.stack([values[index : index + horizon].T for index in indices])
    return TensorDataset(
        torch.from_numpy(histories.astype(np.float32, copy=False)),
        torch.from_numpy(targets.astype(np.float32, copy=False)),
    )


def make_target_windows(
    values: np.ndarray,
    target_start: int,
    target_end: int,
    lag: int,
    horizon: int | None = None,
) -> TensorDataset:
    """按目标起点构造窗口；历史可来自目标块之前，标签必须完整存在。"""
    horizon = HORIZON if horizon is None else int(horizon)
    first = max(int(target_start), int(lag))
    last = min(int(target_end), len(values) - horizon + 1)
    if last <= first:
        raise ValueError(
            f"Target window [{target_start}, {target_end}) has no complete "
            f"window for L={lag}, H={horizon}"
        )
    indices = np.arange(first, last, dtype=np.int64)
    histories = np.stack([values[index - lag : index].T for index in indices])
    targets = np.stack([values[index : index + horizon].T for index in indices])
    return TensorDataset(
        torch.from_numpy(histories.astype(np.float32, copy=False)),
        torch.from_numpy(targets.astype(np.float32, copy=False)),
    )


def empty_dataset_like(dataset: TensorDataset) -> TensorDataset:
    return TensorDataset(dataset.tensors[0][:0], dataset.tensors[1][:0])


def mature_windows(
    values: np.ndarray,
    online_start: int,
    revealed_end: int,
    lag: int,
    horizon: int | None = None,
) -> TensorDataset:
    """只返回标签终点不晚于 revealed_end 的在线窗口。"""
    horizon = HORIZON if horizon is None else int(horizon)
    return make_target_windows(
        values, online_start, revealed_end - horizon + 1, lag, horizon
    )


def reported_steps(horizon: int) -> tuple[int, ...]:
    """报告步长不能超过模型实际输出步长。"""
    candidates = tuple(dict.fromkeys((*REPORT_STEPS, int(horizon))))
    return tuple(step for step in candidates if step <= int(horizon))


def chronological_split(dataset: TensorDataset, fraction: float = 0.80) -> tuple[TensorDataset, TensorDataset]:
    split = min(max(1, int(len(dataset) * fraction)), len(dataset) - 1)
    train = TensorDataset(dataset.tensors[0][:split], dataset.tensors[1][:split])
    validation = TensorDataset(dataset.tensors[0][split:], dataset.tensors[1][split:])
    return train, validation


class MultiHorizonSelectiveGraphForecaster(nn.Module):
    """The SCA decoupled backbone with a direct 12-step output head."""

    def __init__(
        self,
        n_nodes: int,
        lag: int,
        support: np.ndarray,
        horizon: int = HORIZON,
        hidden: int = 24,
        graph_rank: int = 8,
        revin: bool | None = None,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.horizon = horizon
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
        self.temporal_readout = nn.Linear(hidden, horizon)
        self.graph_readout = nn.Linear(hidden, horizon, bias=False)
        time_width = 1 if MODEL_OPTIONS["compact_time"] else horizon
        self.time_scale = nn.Parameter(torch.zeros(n_nodes, time_width))
        self.time_bias = nn.Parameter(torch.zeros(n_nodes, time_width))
        generator = torch.Generator().manual_seed(12345)
        self.graph_u = nn.Parameter(
            torch.randn(n_nodes, graph_rank, generator=generator) * 0.5
        )
        self.graph_v = nn.Parameter(torch.zeros(n_nodes, graph_rank))
        # Optional exploration modules; identity at initialization, absent by default.
        self.revin = bool(MODEL_OPTIONS["revin"]) if revin is None else bool(revin)
        self.weight_mod = bool(MODEL_OPTIONS["weight_mod"])
        self.msg_adapter = bool(MODEL_OPTIONS["msg_adapter"])
        if self.weight_mod:
            rank = int(TUNE_OPTIONS["mod_rank"])
            mod_generator = torch.Generator().manual_seed(54321)
            self.mod_a1 = nn.Parameter(torch.randn(hidden, rank, generator=mod_generator) * 0.1)
            self.mod_b1 = nn.Parameter(torch.zeros(lag, rank))
            self.mod_a2 = nn.Parameter(torch.randn(hidden, rank, generator=mod_generator) * 0.1)
            self.mod_b2 = nn.Parameter(torch.zeros(hidden, rank))
        if self.msg_adapter:
            bottleneck = 4
            msg_generator = torch.Generator().manual_seed(24680)
            self.msg_down = nn.Parameter(torch.randn(hidden, bottleneck, generator=msg_generator) * 0.1)
            self.msg_up = nn.Parameter(torch.zeros(bottleneck, hidden))

    def adjacency(self) -> torch.Tensor:
        delta = 5.0 * (self.graph_u @ self.graph_v.T)
        adjacency = torch.sigmoid(self.base_graph_logits + delta) * self.off_diagonal
        return adjacency / (adjacency.sum(dim=1, keepdim=True) + 1e-6)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        if not self.weight_mod:
            return self.temporal(history)
        first, second = self.temporal[0], self.temporal[2]
        w1 = first.weight * (1.0 + self.mod_a1 @ self.mod_b1.T)
        w2 = second.weight * (1.0 + self.mod_a2 @ self.mod_b2.T)
        hidden = torch.relu(torch.nn.functional.linear(history, w1, first.bias))
        return torch.relu(torch.nn.functional.linear(hidden, w2, second.bias))

    def parts(self, history: torch.Tensor):
        """Adapter-independent pieces in the (optionally instance-normalized) space."""
        mean = std = None
        if self.revin:
            mean = history.mean(dim=-1, keepdim=True)
            std = torch.sqrt(history.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)
            history = (history - mean) / std
        hidden = self.encode(history)
        local_prediction = self.temporal_readout(hidden)
        messages = torch.einsum("ij,bjh->bih", self.adjacency(), hidden)
        if self.msg_adapter:
            messages = messages + torch.relu(messages @ self.msg_down) @ self.msg_up
        graph_prediction = self.graph_readout(messages)
        return local_prediction, graph_prediction, mean, std

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        local_prediction, graph_prediction, mean, std = self.parts(history)
        correction = local_prediction * self.time_scale + self.time_bias
        output = local_prediction + graph_prediction + correction
        if mean is not None:
            output = output * std + mean
        return output


class DLinearMultiHorizon(nn.Module):
    def __init__(self, sequence_length: int, horizon: int | None = None, kernel_size: int = 3) -> None:
        super().__init__()
        horizon = HORIZON if horizon is None else int(horizon)
        self.kernel_size = kernel_size
        self.trend = nn.Linear(sequence_length, horizon)
        self.seasonal = nn.Linear(sequence_length, horizon)

    def moving_average(self, history: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        padded = torch.cat(
            (
                history[..., :1].repeat(1, 1, padding),
                history,
                history[..., -1:].repeat(1, 1, padding),
            ),
            dim=-1,
        )
        return torch.nn.functional.avg_pool1d(
            padded, kernel_size=self.kernel_size, stride=1
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(history)
        seasonal = history - trend
        return self.trend(trend) + self.seasonal(seasonal)


def normalized_support(support: np.ndarray) -> torch.Tensor:
    adjacency = support.astype(np.float32) + np.eye(len(support), dtype=np.float32)
    degree = np.maximum(adjacency.sum(axis=1), 1e-6)
    inverse = np.diag(degree ** -0.5)
    return torch.from_numpy(inverse @ adjacency @ inverse).float()


class STGCNMultiHorizon(nn.Module):
    def __init__(self, support: np.ndarray, lag: int, horizon: int | None = None) -> None:
        super().__init__()
        horizon = HORIZON if horizon is None else int(horizon)
        path = (
            ROOT
            / "external_baselines"
            / "CoMemNet"
            / "baseline"
            / "STKEC-main"
            / "src"
            / "model"
            / "stgcn.py"
        )
        spec = importlib.util.spec_from_file_location("bundled_stgcn_multihorizon", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import STGCN from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.network = module.STGCN(
            gso=normalized_support(support),
            blocks=[[1], [32, 16, 32], [32, 16, 32], [64, 64], [horizon]],
            Kt=3,
            Ks=3,
            dropout=0.1,
            node_num=len(support),
            input_dim=1,
            output_dim=1,
            seq_len=lag,
            horizon=horizon,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        for module in self.network.modules():
            if hasattr(module, "gso"):
                module.gso = module.gso.to(history.device)
        sequence = history.permute(0, 2, 1).unsqueeze(-1)
        output = self.network(sequence)
        return output.squeeze(-1).permute(0, 2, 1)


class PatchTSTMultiHorizon(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        horizon: int | None = None,
        patch_length: int = 4,
        stride: int = 2,
        model_dim: int = 32,
    ) -> None:
        super().__init__()
        horizon = HORIZON if horizon is None else int(horizon)
        self.patch_length = patch_length
        self.stride = stride
        self.horizon = horizon
        patch_count = 1 + (sequence_length - patch_length) // stride
        self.patch_projection = nn.Linear(patch_length, model_dim)
        self.position = nn.Parameter(torch.zeros(1, patch_count, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(patch_count * model_dim, horizon)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = history.shape
        channels = history.reshape(batch * nodes, -1)
        outputs = []
        for channel_chunk in channels.split(4096):
            patches = channel_chunk.unfold(-1, self.patch_length, self.stride)
            encoded = self.patch_projection(patches) + self.position
            encoded = self.encoder(encoded)
            outputs.append(self.head(encoded.flatten(1)))
        return torch.cat(outputs).reshape(batch, nodes, self.horizon)


@torch.no_grad()
def prediction_tensor(model: nn.Module, dataset: TensorDataset, device: torch.device) -> torch.Tensor:
    model.eval()
    outputs = []
    for history, _ in DataLoader(dataset, batch_size=int(MEMORY_OPTIONS["eval_batch"]), shuffle=False):
        outputs.append(model(history.to(device)).cpu())
    return torch.cat(outputs)


@torch.no_grad()
def normalized_mse(model: nn.Module, dataset: TensorDataset, device: torch.device) -> float:
    model.eval()
    squared = 0.0
    count = 0
    for history, target in DataLoader(dataset, batch_size=int(MEMORY_OPTIONS["eval_batch"]), shuffle=False):
        prediction = model(history.to(device))
        target = target.to(device)
        squared += torch.sum((prediction - target) ** 2).item()
        count += target.numel()
    return squared / count


@torch.no_grad()
def metric_sums(
    model: nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float | int]:
    model.eval()
    result: dict[str, float | int] = {
        "normalized_abs_sum": 0.0,
        "normalized_sq_sum": 0.0,
        "normalized_count": 0,
    }
    labels = ["all"] + [f"h{step}" for step in reported_steps(HORIZON)]
    for label in labels:
        result[f"abs_sum_{label}"] = 0.0
        result[f"sq_sum_{label}"] = 0.0
        result[f"ape_sum_{label}"] = 0.0
        result[f"valid_count_{label}"] = 0
    center = torch.from_numpy(mean).to(device).view(1, -1, 1)
    scale = torch.from_numpy(std).to(device).view(1, -1, 1)
    for history, target in DataLoader(dataset, batch_size=int(MEMORY_OPTIONS["eval_batch"]), shuffle=False):
        prediction = model(history.to(device))
        target = target.to(device)
        normalized_error = prediction - target
        result["normalized_abs_sum"] += float(
            torch.sum(torch.abs(normalized_error)).item()
        )
        result["normalized_sq_sum"] += float(torch.sum(normalized_error**2).item())
        result["normalized_count"] += int(target.numel())
        raw_prediction = prediction * scale + center
        raw_target = target * scale + center
        selections = [("all", slice(None))] + [
            (f"h{step}", step - 1) for step in reported_steps(HORIZON)
        ]
        for label, selection in selections:
            pred = raw_prediction[..., selection]
            truth = raw_target[..., selection]
            valid = truth != 0
            error = pred[valid] - truth[valid]
            result[f"abs_sum_{label}"] += float(torch.sum(torch.abs(error)).item())
            result[f"sq_sum_{label}"] += float(torch.sum(error**2).item())
            result[f"ape_sum_{label}"] += float(
                torch.sum(torch.abs(error) / torch.clamp(torch.abs(truth[valid]), min=1e-5)).item()
            )
            result[f"valid_count_{label}"] += int(valid.sum().item())
    return result


def fit_model(
    model: nn.Module,
    train_set: TensorDataset,
    validation_set: TensorDataset,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    selective: bool = False,
) -> nn.Module:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return model
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale = 0
    for _ in range(epochs):
        model.train()
        for history, target in DataLoader(train_set, batch_size=128, shuffle=True):
            history = history.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            micro = int(MEMORY_OPTIONS["micro_batch"])
            if micro <= 0 or micro >= len(history):
                loss = torch.mean((model(history) - target) ** 2)
                if selective:
                    loss = loss + 1e-4 * (
                        torch.mean(model.graph_u**2) + torch.mean(model.graph_v**2)
                    )
                loss.backward()
            else:
                for part_x, part_y in zip(history.split(micro), target.split(micro)):
                    weight = len(part_x) / len(history)
                    (torch.mean((model(part_x) - part_y) ** 2) * weight).backward()
                penalty = 1e-4 * (torch.mean(model.graph_u**2) + torch.mean(model.graph_v**2)) if selective else None
                if penalty is not None and penalty.requires_grad:
                    penalty.backward()
            optimizer.step()
        validation = normalized_mse(model, validation_set, device)
        if validation < best_validation - 1e-7:
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state)
    return model


def estimate_fisher(
    model: nn.Module, dataset: TensorDataset, device: torch.device
) -> dict[str, torch.Tensor]:
    fisher = {
        name: torch.zeros_like(parameter, device=device)
        for name, parameter in model.named_parameters()
    }
    batches = 0
    model.train()
    for history, target in DataLoader(dataset, batch_size=128, shuffle=False):
        history = history.to(device)
        target = target.to(device)
        model.zero_grad(set_to_none=True)
        torch.mean((model(history) - target) ** 2).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                fisher[name] += parameter.grad.detach() ** 2
        batches += 1
    for name in fisher:
        fisher[name] /= max(batches, 1)
    model.zero_grad(set_to_none=True)
    return fisher


def fit_ewc(
    model: MultiHorizonSelectiveGraphForecaster,
    train_set: TensorDataset,
    validation_set: TensorDataset,
    historical_set: TensorDataset,
    device: torch.device,
    epochs: int,
    regularization: float,
) -> MultiHorizonSelectiveGraphForecaster:
    configure_action(model, "full")
    anchors = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    fisher = estimate_fisher(model, historical_set, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale = 0
    for _ in range(epochs):
        model.train()
        for history, target in DataLoader(train_set, batch_size=128, shuffle=True):
            history = history.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            penalty = sum(
                torch.sum(fisher[name] * (parameter - anchors[name]) ** 2)
                for name, parameter in model.named_parameters()
            )
            loss = torch.mean((model(history) - target) ** 2) + regularization * penalty
            loss.backward()
            optimizer.step()
        validation = normalized_mse(model, validation_set, device)
        if validation < best_validation - 1e-7:
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state)
    return model


@dataclass
class DERBuffer:
    history: torch.Tensor
    target: torch.Tensor
    prediction: torch.Tensor


def make_der_buffer(
    model: nn.Module,
    dataset: TensorDataset,
    size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> DERBuffer:
    sampled = sample_dataset(dataset, size, rng)
    return DERBuffer(
        sampled.tensors[0],
        sampled.tensors[1],
        prediction_tensor(model, sampled, device),
    )


def refresh_der_buffer(
    buffer: DERBuffer,
    current: TensorDataset,
    current_prediction: torch.Tensor,
    size: int,
    rng: np.random.Generator,
) -> DERBuffer:
    history = torch.cat((buffer.history, current.tensors[0]))
    target = torch.cat((buffer.target, current.tensors[1]))
    prediction = torch.cat((buffer.prediction, current_prediction))
    if len(history) <= size:
        return DERBuffer(history, target, prediction)
    indices = torch.from_numpy(rng.choice(len(history), size=size, replace=False))
    return DERBuffer(history[indices], target[indices], prediction[indices])


def fit_derpp(
    model: MultiHorizonSelectiveGraphForecaster,
    train_set: TensorDataset,
    validation_set: TensorDataset,
    buffer: DERBuffer,
    device: torch.device,
    epochs: int,
    alpha: float,
    beta: float,
) -> MultiHorizonSelectiveGraphForecaster:
    configure_action(model, "full")
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)
    replay_set = TensorDataset(buffer.history, buffer.target, buffer.prediction)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale = 0
    for _ in range(epochs):
        model.train()
        replay_iterator = iter(DataLoader(replay_set, batch_size=128, shuffle=True))
        for history, target in DataLoader(train_set, batch_size=128, shuffle=True):
            try:
                replay_history, replay_target, replay_prediction = next(replay_iterator)
            except StopIteration:
                replay_iterator = iter(DataLoader(replay_set, batch_size=128, shuffle=True))
                replay_history, replay_target, replay_prediction = next(replay_iterator)
            history = history.to(device)
            target = target.to(device)
            replay_history = replay_history.to(device)
            replay_target = replay_target.to(device)
            replay_prediction = replay_prediction.to(device)
            optimizer.zero_grad(set_to_none=True)
            current_loss = torch.mean((model(history) - target) ** 2)
            replay_output = model(replay_history)
            loss = (
                current_loss
                + alpha * torch.mean((replay_output - replay_prediction) ** 2)
                + beta * torch.mean((replay_output - replay_target) ** 2)
            )
            loss.backward()
            optimizer.step()
        validation = normalized_mse(model, validation_set, device)
        if validation < best_validation - 1e-7:
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def solve_time_adapter(
    model: MultiHorizonSelectiveGraphForecaster,
    train_set: TensorDataset,
    device: torch.device,
    ridge: float,
) -> None:
    """Closed-form ridge solution of the per-node/horizon scale and bias.

    The adapter-independent part of the prediction is held fixed, so each series
    solves a 2x2 normal system; the penalty shrinks toward the current state.
    """
    model.eval()
    nodes, horizon = model.n_nodes, model.horizon
    totals = [torch.zeros(nodes, horizon, device=device) for _ in range(4)]
    count = 0
    for history, target in DataLoader(train_set, batch_size=256, shuffle=False):
        history = history.to(device)
        target = target.to(device)
        local, graph, mean, std = model.parts(history)
        if mean is not None:
            target = (target - mean) / std
        residual = target - (local + graph)
        totals[0] += (local * local).sum(dim=0)
        totals[1] += local.sum(dim=0)
        totals[2] += (local * residual).sum(dim=0)
        totals[3] += residual.sum(dim=0)
        count += local.shape[0]
    width = model.time_scale.shape[1]
    sll, sl, slr, sr = (
        (t.sum(dim=1, keepdim=True) if width == 1 else t) for t in totals
    )
    samples = float(count * (horizon if width == 1 else 1))
    scale0, bias0 = model.time_scale.data, model.time_bias.data
    a11 = sll + ridge
    a12 = sl
    a22 = samples + ridge
    rhs1 = slr + ridge * scale0
    rhs2 = sr + ridge * bias0
    det = a11 * a22 - a12 * a12
    model.time_scale.data.copy_((rhs1 * a22 - a12 * rhs2) / det)
    model.time_bias.data.copy_((a11 * rhs2 - a12 * rhs1) / det)


def fit_adapters(
    model: MultiHorizonSelectiveGraphForecaster,
    action: str,
    train_set: TensorDataset,
    validation_set: TensorDataset,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> MultiHorizonSelectiveGraphForecaster:
    """v1 gradient fit, or (optionally) a closed-form time-adapter solution."""
    if not FIT_OPTIONS["closed_form_time"] or action not in ("time", "joint"):
        return fit_model(
            model, train_set, validation_set, device, epochs, learning_rate, selective=True
        )
    ridge = float(FIT_OPTIONS["ridge"])
    initial_state = copy.deepcopy(model.state_dict())
    initial_validation = normalized_mse(model, validation_set, device)
    solve_time_adapter(model, train_set, device, ridge)
    model.time_scale.requires_grad = False
    model.time_bias.requires_grad = False
    if any(parameter.requires_grad for parameter in model.parameters()):
        model = fit_model(
            model, train_set, validation_set, device, epochs, learning_rate, selective=True
        )
        solve_time_adapter(model, train_set, device, ridge)
    # Never keep an update that is worse than the starting state on held-out data.
    if normalized_mse(model, validation_set, device) > initial_validation:
        model.load_state_dict(initial_state)
    return model


ROUTE_OPTIONS = {"variant": "full", "no_memory": False}
ROUTE_VARIANTS = ("full", "no_effect_gate", "no_positivity_gate", "marginal_only",
                  "dependency_only", "always_joint")


def apply_route_variant(diagnosis: dict, minimum_effect_z: float, alpha: float) -> str:
    """Re-derive the routed action under a routing ablation variant."""
    variant = ROUTE_OPTIONS["variant"]
    if variant == "full":
        return str(diagnosis["routed_action"])
    if variant == "always_joint":
        return "joint"
    effect = variant != "no_effect_gate"
    marginal = float(diagnosis["p_marginal"]) <= alpha and (
        not effect or float(diagnosis["z_marginal"]) >= minimum_effect_z)
    dependency = float(diagnosis["p_dependency"]) <= alpha and (
        not effect or float(diagnosis["z_dependency"]) >= minimum_effect_z)
    if variant != "no_positivity_gate" and float(diagnosis["score_dependency"]) <= 0.0:
        dependency = False
    if variant == "marginal_only":
        dependency = False
    elif variant == "dependency_only":
        marginal = False
    if marginal and dependency:
        return "joint"
    if marginal:
        return "time"
    if dependency:
        return "graph"
    return "freeze"


class MechanismMemory:
    def __init__(self, base_model: MultiHorizonSelectiveGraphForecaster, device: torch.device) -> None:
        self.base_model = copy.deepcopy(base_model).to(device)
        self.device = device
        self.memory: dict[str, dict[str, torch.Tensor]] = {}
        self.active_action = "freeze"
        self.active_model = self.compose("freeze")

    @staticmethod
    def adapter_state(model: MultiHorizonSelectiveGraphForecaster) -> dict[str, torch.Tensor]:
        return {
            name: getattr(model, name).detach().cpu().clone()
            for name in ADAPTER_NAMES
            if hasattr(model, name)
        }

    @staticmethod
    def load_adapter_state(
        model: MultiHorizonSelectiveGraphForecaster,
        state: dict[str, torch.Tensor],
    ) -> None:
        for name, value in state.items():
            getattr(model, name).data.copy_(value.to(getattr(model, name).device))

    def compose(self, action: str) -> MultiHorizonSelectiveGraphForecaster:
        model = copy.deepcopy(self.base_model).to(self.device)
        if ROUTE_OPTIONS["no_memory"] and "shared" in self.memory:
            self.load_adapter_state(model, self.memory["shared"])
        elif action in self.memory:
            self.load_adapter_state(model, self.memory[action])
        elif action == "joint":
            if "time" in self.memory:
                for name in TIME_NAMES:
                    if name in self.memory["time"]:
                        getattr(model, name).data.copy_(self.memory["time"][name].to(self.device))
            if "graph" in self.memory:
                for name in GRAPH_NAMES:
                    if name in self.memory["graph"]:
                        getattr(model, name).data.copy_(self.memory["graph"][name].to(self.device))
        return model

    def adapt(
        self,
        action: str,
        train_set: TensorDataset,
        validation_set: TensorDataset,
        epochs: int,
        random_seed: int,
    ) -> tuple[int, float]:
        set_seed(random_seed)
        model = self.compose(action)
        trainable = configure_action(model, action)
        started = time.perf_counter()
        if trainable:
            model = fit_adapters(
                model,
                action,
                train_set,
                validation_set,
                self.device,
                epochs,
                action_learning_rate(action),
            )
        elapsed = time.perf_counter() - started
        if action != "freeze":
            self.memory["shared" if ROUTE_OPTIONS["no_memory"] else action] = self.adapter_state(model)
        self.active_action = action
        self.active_model = model
        return trainable, elapsed

    def candidate(
        self,
        action: str,
        train_set: TensorDataset,
        validation_set: TensorDataset,
        epochs: int,
        random_seed: int,
    ) -> tuple[MultiHorizonSelectiveGraphForecaster, int, float, float]:
        """Fit an action without committing it, and report its validation MSE."""
        set_seed(random_seed)
        model = self.compose(action)
        trainable = configure_action(model, action)
        started = time.perf_counter()
        if trainable:
            model = fit_adapters(
                model,
                action,
                train_set,
                validation_set,
                self.device,
                epochs,
                action_learning_rate(action),
            )
        elapsed = time.perf_counter() - started
        return model, trainable, elapsed, normalized_mse(model, validation_set, self.device)

    def commit(self, action: str, model: MultiHorizonSelectiveGraphForecaster) -> None:
        if action != "freeze":
            self.memory["shared" if ROUTE_OPTIONS["no_memory"] else action] = self.adapter_state(model)
        self.active_action = action
        self.active_model = model


def adapt_with_upgrade(
    memory: "MechanismMemory",
    routed_action: str,
    train_set: TensorDataset,
    validation_set: TensorDataset,
    epochs: int,
    random_seed: int,
    tau: float,
) -> tuple[str, int, float, dict[str, object]]:
    """Diagnosis proposes an action; validation may upgrade it to ``joint``.

    Only the adopted candidate is committed to memory. The joint candidate is
    adopted when its validation MSE is at least ``tau`` (relative) below that of
    the routed action. The extra candidate's time is included in the elapsed time.
    """
    routed_model, routed_trainable, routed_time, routed_val = memory.candidate(
        routed_action, train_set, validation_set, epochs, random_seed
    )
    if routed_action == "joint":
        memory.commit(routed_action, routed_model)
        return routed_action, routed_trainable, routed_time, {
            "upgraded": False, "val_routed": routed_val, "val_joint": routed_val,
        }
    joint_model, joint_trainable, joint_time, joint_val = memory.candidate(
        "joint", train_set, validation_set, epochs, random_seed + 1
    )
    elapsed = routed_time + joint_time
    if joint_val <= routed_val * (1.0 - tau):
        memory.commit("joint", joint_model)
        return "joint", joint_trainable, elapsed, {
            "upgraded": True, "val_routed": routed_val, "val_joint": joint_val,
        }
    memory.commit(routed_action, routed_model)
    return routed_action, routed_trainable, elapsed, {
        "upgraded": False, "val_routed": routed_val, "val_joint": joint_val,
    }


def adapt_selective_copy(
    model: MultiHorizonSelectiveGraphForecaster,
    action: str,
    train_set: TensorDataset,
    validation_set: TensorDataset,
    device: torch.device,
    epochs: int,
    random_seed: int,
) -> tuple[MultiHorizonSelectiveGraphForecaster, int, float]:
    set_seed(random_seed)
    candidate = copy.deepcopy(model).to(device)
    trainable = configure_action(candidate, action)
    started = time.perf_counter()
    if trainable:
        candidate = fit_model(
            candidate,
            train_set,
            validation_set,
            device,
            epochs,
            action_learning_rate(action),
            selective=True,
        )
    return candidate, trainable, time.perf_counter() - started


def method_category(method: str) -> str:
    if method == "Frozen" or method in FORECAST_CONTROLS:
        return "forecast_control"
    if method in GENERIC_CL:
        return "generic_continual_learning"
    return "specialized_continual_learning"


def load_stream_data(
    path: Path,
    warmup_ratio: float,
    feature_index: int,
    max_rows: int | None,
    max_nodes: int | None,
    warmup_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        raw = payload["data"][:, :, feature_index].astype(np.float32)
        source_metadata: dict[str, object] = {
            "source_format": "npz",
            "source_key": "data",
            "feature_index": feature_index,
        }
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError(f"No numeric columns found in {path}")
        raw = numeric.to_numpy(dtype=np.float32)
        source_metadata = {
            "source_format": "csv",
            "numeric_columns": numeric.columns.tolist(),
        }
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    if warmup_only:
        # Validation stream: only the warm-up segment of the full stream, which is itself
        # split into an inner warm-up and online blocks. The real online stream is never read.
        raw = raw[: int(len(raw) * warmup_ratio)]
    if max_rows is not None:
        raw = raw[:max_rows]
    if max_nodes is not None:
        raw = raw[:, :max_nodes]
    warmup_end = int(len(raw) * warmup_ratio)
    mean = raw[:warmup_end].mean(axis=0).astype(np.float32)
    std = raw[:warmup_end].std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    values = ((raw - mean) / std).astype(np.float32)
    metadata = {
        "source": str(path.resolve()),
        **source_metadata,
        "rows": int(len(raw)),
        "nodes": int(raw.shape[1]),
        "warmup_end": warmup_end,
        "normalization": "per-node mean/std estimated from the warm-up only",
    }
    return values, mean, std, metadata


def evaluation_models(
    models: dict[str, nn.Module], memories: dict[str, MechanismMemory]
) -> dict[str, nn.Module]:
    return {**models, **{name: memory.active_model for name, memory in memories.items()}}


class GatedMixture(nn.Module):
    """Prediction = base + gate * (adapted - base), with a per-node gate in [0, 1]."""

    def __init__(self, base: nn.Module, adapted: nn.Module, gate: torch.Tensor) -> None:
        super().__init__()
        self.base = base
        self.adapted = adapted
        self.register_buffer("gate", gate)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        base_prediction = self.base(history)
        return base_prediction + self.gate * (self.adapted(history) - base_prediction)


@torch.no_grad()
def node_mse(model: nn.Module, dataset: TensorDataset, device: torch.device) -> torch.Tensor:
    model.eval()
    total = None
    count = 0
    for history, target in DataLoader(dataset, batch_size=int(MEMORY_OPTIONS["eval_batch"]), shuffle=False):
        error = (model(history.to(device)) - target.to(device)) ** 2
        part = error.sum(dim=(0, 2))
        total = part if total is None else total + part
        count += error.shape[0] * error.shape[2]
    return total / max(count, 1)


def update_gate(
    gate: torch.Tensor, adapted_mse: torch.Tensor, base_mse: torch.Tensor, eta: float
) -> torch.Tensor:
    """Exponentiated-gradient update of the weight on the adapted model."""
    scale = torch.clamp((adapted_mse + base_mse) / 2.0, min=1e-8)
    weight_adapted = gate.view(-1) * torch.exp(-eta * adapted_mse / scale)
    weight_base = (1.0 - gate.view(-1)) * torch.exp(-eta * base_mse / scale)
    updated = weight_adapted / (weight_adapted + weight_base)
    return torch.clamp(updated, 0.05, 0.95).view(-1, 1)


def build_view_b(
    base_a: MultiHorizonSelectiveGraphForecaster,
    base_train: TensorDataset,
    base_validation: TensorDataset,
    device: torch.device,
    epochs: int,
    seed: int,
) -> MultiHorizonSelectiveGraphForecaster:
    """Instance-normalized view: initialized from A, graph logits shared, temporal parts retrained."""
    set_seed(seed + 40_000)
    view = copy.deepcopy(base_a)
    view.revin = True
    configure_action(view, "base")
    view.base_graph_logits.requires_grad = False
    return fit_model(view, base_train, base_validation, device, epochs, 2e-3, selective=True)


def view_b_extra_parameters(view: MultiHorizonSelectiveGraphForecaster) -> int:
    return sum(
        parameter.numel()
        for module in (view.temporal, view.temporal_readout, view.graph_readout)
        for parameter in module.parameters()
    )


class DualViewMemory:
    """Plain view A and normalized view B, each with its own action-keyed memory.

    The diagnosed action is applied to the active view only. After each block, the
    realized errors of both views decide the view used for the next block.
    """

    def __init__(self, base_a, base_b, device: torch.device) -> None:
        self.views = {"A": MechanismMemory(base_a, device), "B": MechanismMemory(base_b, device)}
        self.view = "A"
        self.base_model = self.views["A"].base_model
        self.device = device

    @property
    def active_model(self) -> nn.Module:
        return self.views[self.view].active_model

    @property
    def active_action(self) -> str:
        return self.views[self.view].active_action

    def select_view(
        self, observed: TensorDataset, routed_action: str, delta: float
    ) -> dict[str, object]:
        error_a = normalized_mse(self.views["A"].active_model, observed, self.device)
        error_b = normalized_mse(self.views["B"].active_model, observed, self.device)
        previous = self.view
        if self.view == "A":
            if routed_action in ("time", "joint") and error_b < (1.0 - delta) * error_a:
                self.view = "B"
        elif error_a < (1.0 - delta) * error_b:
            self.view = "A"
        return {
            "view_error_a": error_a,
            "view_error_b": error_b,
            "view_next": self.view,
            "view_switched": self.view != previous,
        }

    def adapt(self, action, train_set, validation_set, epochs, random_seed):
        return self.views[self.view].adapt(action, train_set, validation_set, epochs, random_seed)


class EnsembleMemory:
    """Adapted view A averaged with a frozen view B using exponentiated-gradient weights."""

    def __init__(self, base_a, frozen_b, device: torch.device) -> None:
        self.memory = MechanismMemory(base_a, device)
        self.frozen_b = copy.deepcopy(frozen_b).to(device)
        self.weight = torch.full((1, 1), 0.5, device=device)
        self.base_model = self.memory.base_model
        self.device = device

    @property
    def active_model(self) -> nn.Module:
        return GatedMixture(self.frozen_b, self.memory.active_model, self.weight)

    @property
    def active_action(self) -> str:
        return self.memory.active_action

    def update_weight(self, observed: TensorDataset, eta: float) -> dict[str, object]:
        error_a = normalized_mse(self.memory.active_model, observed, self.device)
        error_b = normalized_mse(self.frozen_b, observed, self.device)
        self.weight = update_gate(
            self.weight,
            torch.tensor([error_a], device=self.device),
            torch.tensor([error_b], device=self.device),
            eta,
        )
        return {
            "view_error_a": error_a,
            "view_error_b": error_b,
            "weight_on_a": float(self.weight.item()),
        }

    def adapt(self, action, train_set, validation_set, epochs, random_seed):
        return self.memory.adapt(action, train_set, validation_set, epochs, random_seed)


def run_seed(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    warmup_end: int,
    seed: int,
    device: torch.device,
    lag: int,
    online_blocks: int,
    base_epochs: int,
    adapt_epochs: int,
    null_windows: int,
    alpha: float,
    uncertain_alpha: float,
    minimum_effect_z: float,
    replay_size: int,
    graph_top_k: int,
    ewc_lambda: float,
    der_alpha: float,
    der_beta: float,
    selected_methods: tuple[str, ...],
    split_mode: str = "legacy",
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if split_mode == "proceed":
        base_train_end = int(warmup_end * 0.80)
        base_validation_end = warmup_end
    else:
        base_train_end = int(warmup_end * 0.70)
        base_validation_end = int(warmup_end * 0.85)
    base_train = make_windows(values, 0, base_train_end, lag)
    base_validation = make_windows(values, base_train_end, base_validation_end, lag)
    historical_test = (
        base_validation
        if split_mode == "proceed"
        else make_windows(values, base_validation_end, warmup_end, lag)
    )
    support = infer_support(values[:warmup_end], graph_top_k)
    block_edges = np.linspace(warmup_end, len(values), online_blocks + 1, dtype=int)
    # 校准窗口只能来自训练段，不能因为在线块较大而倒灌在线数据。
    calibration_window = min(
        int(np.min(np.diff(block_edges))),
        max(lag + 1, warmup_end - lag - 1),
    )
    calibration = calibrate_natural_null(
        values,
        warmup_end,
        lag,
        calibration_window,
        null_windows,
        alpha,
        seed,
        support=support,
        diag_lag=min(lag, 12),
    )

    set_seed(seed)
    base = MultiHorizonSelectiveGraphForecaster(
        values.shape[1], lag, support, horizon=HORIZON
    ).to(device)
    configure_action(base, "base")
    base = fit_model(base, base_train, base_validation, device, base_epochs, 2e-3, selective=True)

    models: dict[str, nn.Module] = {}
    for method in ("Frozen", "Online-Full", "ER", "EWC", "DER++", "DSGA-NoMemory"):
        if method in selected_methods:
            models[method] = copy.deepcopy(base)
    if "DLinear-Online" in selected_methods:
        set_seed(seed + 10_000)
        models["DLinear-Online"] = fit_model(
            DLinearMultiHorizon(lag).to(device),
            base_train,
            base_validation,
            device,
            base_epochs,
            1e-3,
        )
    if "STGCN-Online" in selected_methods:
        set_seed(seed + 20_000)
        models["STGCN-Online"] = fit_model(
            STGCNMultiHorizon(support, lag).to(device),
            base_train,
            base_validation,
            device,
            base_epochs,
            1e-3,
        )
    if "PatchTST-Online" in selected_methods:
        set_seed(seed + 30_000)
        models["PatchTST-Online"] = fit_model(
            PatchTSTMultiHorizon(lag).to(device),
            base_train,
            base_validation,
            device,
            base_epochs,
            1e-3,
        )
    memory_methods = tuple(
        method
        for method in ("Always-Time", "Always-Graph", "Always-Joint", "DSGA")
        if method in selected_methods
    )
    memories = {method: MechanismMemory(base, device) for method in memory_methods}
    dv_mode = str(DV_OPTIONS["mode"])
    view_b = None
    if dv_mode != "off" and "DSGA" in selected_methods:
        view_b = build_view_b(base, base_train, base_validation, device, base_epochs, seed)
        memories["DSGA"] = (
            DualViewMemory(base, view_b, device)
            if dv_mode == "ftl"
            else EnsembleMemory(base, view_b, device)
        )
    full_parameters = {
        method: sum(parameter.numel() for parameter in model.parameters())
        for method, model in evaluation_models(models, memories).items()
    }
    frozen_params_total = sum(parameter.numel() for parameter in base.parameters())
    if view_b is not None:
        full_parameters["DSGA"] = frozen_params_total
        frozen_params_total += view_b_extra_parameters(view_b)
    rng = np.random.default_rng(70_000 + seed)
    replay = sample_dataset(base_train, replay_size, rng)
    der_buffer = (
        make_der_buffer(base, replay, replay_size, rng, device)
        if "DER++" in selected_methods
        else None
    )

    initial_history = {
        method: normalized_mse(model, historical_test, device)
        for method, model in evaluation_models(models, memories).items()
    }
    block_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    gate_enabled = bool(GATE_OPTIONS["enabled"]) and "DSGA" in selected_methods
    gate = torch.full((values.shape[1], 1), 0.5, device=device)
    released_target = warmup_end
    for block_index, (block_start, block_end) in enumerate(
        zip(block_edges[:-1], block_edges[1:], strict=True)
    ):
        # 预测块的目标允许跨到下一块；只有块末揭晓后才进入 matured。
        observed = make_target_windows(values, int(block_start), int(block_end), lag)
        try:
            mature_start = max(warmup_end, released_target)
            train_set = mature_windows(values, mature_start, int(block_end), lag)
        except ValueError:
            train_set = empty_dataset_like(observed)
        # 视图切换、门控只能使用标签已全部揭晓的窗口；H>1 时块尾窗口的目标跨入下一块，不能使用。
        revealed = train_set
        if len(train_set) >= 2:
            train_set, validation_set = chronological_split(train_set)
        else:
            validation_set = empty_dataset_like(train_set)
        rows_by_method: dict[str, dict[str, object]] = {}
        current_models = evaluation_models(models, memories)
        if gate_enabled:
            current_models["DSGA"] = GatedMixture(
                memories["DSGA"].base_model, memories["DSGA"].active_model, gate
            )
        for method in selected_methods:
            sums = metric_sums(current_models[method], observed, device, mean, std)
            row = {
                "seed": seed,
                "block": block_index,
                "block_start": int(block_start),
                "block_end": int(block_end),
                "method": method,
                "category": method_category(method),
                "action_used_for_prediction": (
                    memories[method].active_action if method in memories else "full"
                ),
                **sums,
                "update_action": "none",
                "trainable_parameters": 0,
                "trainable_fraction": 0.0,
                "adaptation_seconds": 0.0,
            }
            rows_by_method[method] = row
            block_rows.append(row)

        if gate_enabled and len(revealed):
            # Realized errors of the model that just predicted this block update the gate.
            gate = update_gate(
                gate,
                node_mse(memories["DSGA"].active_model, revealed, device),
                node_mse(memories["DSGA"].base_model, revealed, device),
                float(GATE_OPTIONS["eta"]),
            )
            rows_by_method["DSGA"]["gate_mean"] = float(gate.mean().item())

        diagnosis = diagnose_stage(
            values[int(block_start) : int(block_end)],
            calibration,
            alpha,
            uncertain_alpha,
            minimum_effect_z,
        )
        routed_action = apply_route_variant(diagnosis, minimum_effect_z, alpha)

        for index, (method, action) in enumerate(ACTION_FOR_FIXED.items()):
            if method not in selected_methods:
                continue
            if len(train_set) < 2:
                continue
            trainable, elapsed = memories[method].adapt(
                action,
                train_set,
                validation_set,
                adapt_epochs,
                seed * 100_000 + block_index * 100 + index,
            )
            rows_by_method[method].update(
                update_action=action,
                trainable_parameters=trainable,
                trainable_fraction=trainable / full_parameters[method],
                adaptation_seconds=elapsed,
            )

        dv_info: dict[str, object] = {}
        if "DSGA" in selected_methods and view_b is not None and len(revealed):
            if dv_mode == "ftl":
                dv_info = memories["DSGA"].select_view(
                    revealed, routed_action, float(DV_OPTIONS["delta"])
                )
            else:
                dv_info = memories["DSGA"].update_weight(revealed, float(DV_OPTIONS["eta"]))
            dv_info["frozen_params_total"] = frozen_params_total

        if "DSGA" in selected_methods:
            applied_action = "freeze" if dv_mode == "eg_frozen" else routed_action
            extra: dict[str, object] = {}
            if len(train_set) < 2:
                continue
            if UPGRADE_OPTIONS["tau"] is None:
                trainable, elapsed = memories["DSGA"].adapt(
                    applied_action,
                    train_set,
                    validation_set,
                    adapt_epochs,
                    seed * 100_000 + block_index * 100 + 10,
                )
            else:
                applied_action, trainable, elapsed, extra = adapt_with_upgrade(
                    memories["DSGA"],
                    routed_action,
                    train_set,
                    validation_set,
                    adapt_epochs,
                    seed * 100_000 + block_index * 100 + 10,
                    float(UPGRADE_OPTIONS["tau"]),
                )
            gate_parameters = int(values.shape[1]) if gate_enabled else 0
            rows_by_method["DSGA"].update(
                update_action=applied_action,
                trainable_parameters=trainable + gate_parameters,
                trainable_fraction=(trainable + gate_parameters) / full_parameters["DSGA"],
                adaptation_seconds=elapsed,
                routed_action=routed_action,
                **extra,
                **dv_info,
            )

        if "DSGA-NoMemory" in selected_methods:
            if len(train_set) < 2:
                continue
            candidate, trainable, elapsed = adapt_selective_copy(
                models["DSGA-NoMemory"],
                routed_action,
                train_set,
                validation_set,
                device,
                adapt_epochs,
                seed * 100_000 + block_index * 100 + 20,
            )
            models["DSGA-NoMemory"] = candidate
            rows_by_method["DSGA-NoMemory"].update(
                update_action=routed_action,
                trainable_parameters=trainable,
                trainable_fraction=trainable / full_parameters["DSGA-NoMemory"],
                adaptation_seconds=elapsed,
            )

        if "Online-Full" in selected_methods:
            set_seed(seed * 100_000 + block_index * 100 + 30)
            started = time.perf_counter()
            candidate = copy.deepcopy(models["Online-Full"]).to(device)
            configure_action(candidate, "full")
            if len(train_set) < 2:
                continue
            models["Online-Full"] = fit_model(
                candidate, train_set, validation_set, device, adapt_epochs, 8e-4 * float(TUNE_OPTIONS["baseline_lr_scale"]), selective=True
            )
            rows_by_method["Online-Full"].update(
                update_action="full",
                trainable_parameters=full_parameters["Online-Full"],
                trainable_fraction=1.0,
                adaptation_seconds=time.perf_counter() - started,
            )

        if "ER" in selected_methods:
            set_seed(seed * 100_000 + block_index * 100 + 40)
            started = time.perf_counter()
            candidate = copy.deepcopy(models["ER"]).to(device)
            configure_action(candidate, "full")
            if len(train_set) < 2:
                continue
            models["ER"] = fit_model(
                candidate,
                combine_datasets(train_set, replay),
                validation_set,
                device,
                adapt_epochs,
                8e-4 * float(TUNE_OPTIONS["baseline_lr_scale"]),
                selective=True,
            )
            rows_by_method["ER"].update(
                update_action="full+replay",
                trainable_parameters=full_parameters["ER"],
                trainable_fraction=1.0,
                adaptation_seconds=time.perf_counter() - started,
            )

        if "EWC" in selected_methods:
            set_seed(seed * 100_000 + block_index * 100 + 50)
            started = time.perf_counter()
            candidate = copy.deepcopy(models["EWC"]).to(device)
            if len(train_set) < 2:
                continue
            models["EWC"] = fit_ewc(
                candidate,
                train_set,
                validation_set,
                replay,
                device,
                adapt_epochs,
                ewc_lambda,
            )
            rows_by_method["EWC"].update(
                update_action="full+ewc",
                trainable_parameters=full_parameters["EWC"],
                trainable_fraction=1.0,
                adaptation_seconds=time.perf_counter() - started,
            )

        if "DER++" in selected_methods:
            assert der_buffer is not None
            if len(train_set) < 2:
                continue
            der_prediction = prediction_tensor(models["DER++"], train_set, device)
            set_seed(seed * 100_000 + block_index * 100 + 60)
            started = time.perf_counter()
            candidate = copy.deepcopy(models["DER++"]).to(device)
            models["DER++"] = fit_derpp(
                candidate,
                train_set,
                validation_set,
                der_buffer,
                device,
                adapt_epochs,
                der_alpha,
                der_beta,
            )
            rows_by_method["DER++"].update(
                update_action="full+replay+distillation",
                trainable_parameters=full_parameters["DER++"],
                trainable_fraction=1.0,
                adaptation_seconds=time.perf_counter() - started,
            )
            der_buffer = refresh_der_buffer(
                der_buffer, train_set, der_prediction, replay_size, rng
            )

        for index, method in enumerate(FORECAST_CONTROLS):
            if method not in selected_methods:
                continue
            set_seed(seed * 100_000 + block_index * 100 + 70 + index)
            started = time.perf_counter()
            if len(train_set) < 2:
                continue
            models[method] = fit_model(
                models[method],
                train_set,
                validation_set,
                device,
                adapt_epochs,
                5e-4,
            )
            rows_by_method[method].update(
                update_action="online-full",
                trainable_parameters=full_parameters[method],
                trainable_fraction=1.0,
                adaptation_seconds=time.perf_counter() - started,
            )

        if len(train_set):
            replay = sample_dataset(combine_datasets(replay, train_set), replay_size, rng)
        # 下一块补入本块末才成熟的跨块标签窗口。
        released_target = max(released_target, int(block_end) - HORIZON + 1)
        route_rows.append(
            {
                "seed": seed,
                "block": block_index,
                "block_start": int(block_start),
                "block_end": int(block_end),
                "decision_applies_to": block_index + 1,
                **diagnosis,
                "routed_action": routed_action,
            }
        )
        print(
            f"seed={seed} block={block_index + 1}/{online_blocks} route={routed_action}",
            flush=True,
        )

    final_rows = []
    for method, model in evaluation_models(models, memories).items():
        final_mse = normalized_mse(model, historical_test, device)
        final_rows.append(
            {
                "seed": seed,
                "method": method,
                "initial_history_mse": initial_history[method],
                "final_history_mse": final_mse,
                "forgetting": max(final_mse - initial_history[method], 0.0),
            }
        )
    return block_rows, route_rows, final_rows


def summarize(
    block_results: pd.DataFrame,
    final_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    sum_columns = [
        column
        for column in block_results.columns
        if column.startswith(("abs_sum_", "sq_sum_", "ape_sum_", "valid_count_"))
    ] + ["normalized_abs_sum", "normalized_sq_sum", "normalized_count"]
    for (seed, method), group in block_results.groupby(["seed", "method"], sort=False):
        totals = group[sum_columns].sum()
        history = final_history[
            (final_history.seed == seed) & (final_history.method == method)
        ].iloc[0]
        row: dict[str, object] = {
            "seed": int(seed),
            "method": method,
            "category": group.iloc[0].category,
            "prequential_mae": totals.normalized_abs_sum / totals.normalized_count,
            "prequential_mse": totals.normalized_sq_sum / totals.normalized_count,
            "adaptation_seconds": float(group.adaptation_seconds.sum()),
            "mean_trainable_fraction": float(group.trainable_fraction.mean()),
            "forgetting": float(history.forgetting),
        }
        labels = ["all"] + [f"h{step}" for step in reported_steps(HORIZON)]
        for label in labels:
            count = float(totals[f"valid_count_{label}"])
            row[f"mae_{label}"] = float(totals[f"abs_sum_{label}"] / count)
            row[f"rmse_{label}"] = float(np.sqrt(totals[f"sq_sum_{label}"] / count))
            row[f"mape_{label}"] = float(100.0 * totals[f"ape_sum_{label}"] / count)
        seed_rows.append(row)
    seed_summary = pd.DataFrame(seed_rows)
    metric_columns = [
        "prequential_mae",
        "prequential_mse",
        "mae_all",
        "rmse_all",
        "mape_all",
        "forgetting",
        "adaptation_seconds",
        "mean_trainable_fraction",
    ]
    for step in reported_steps(HORIZON):
        metric_columns.extend([f"mae_h{step}", f"rmse_h{step}", f"mape_h{step}"])
    aggregate_rows = []
    for (method, category), group in seed_summary.groupby(["method", "category"], sort=False):
        row: dict[str, object] = {
            "method": method,
            "category": category,
            "seeds": int(group.seed.nunique()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        aggregate_rows.append(row)
    method_summary = pd.DataFrame(aggregate_rows).sort_values("prequential_mae_mean")

    tests = []
    current = seed_summary[seed_summary.method == "DSGA"].sort_values("seed")
    if len(current):
        for method in seed_summary.method.unique():
            if method == "DSGA":
                continue
            baseline = seed_summary[seed_summary.method == method].sort_values("seed")
            merged = current.merge(baseline, on="seed", suffixes=("_sca", "_baseline"))
            if len(merged) < 2:
                continue
            difference = merged.prequential_mae_sca - merged.prequential_mae_baseline
            t_result = stats.ttest_rel(
                merged.prequential_mae_sca, merged.prequential_mae_baseline
            )
            try:
                w_result = stats.wilcoxon(
                    merged.prequential_mae_sca, merged.prequential_mae_baseline
                )
                w_pvalue = float(w_result.pvalue)
            except ValueError:
                w_pvalue = 1.0
            tests.append(
                {
                    "current_method": "DSGA",
                    "baseline": method,
                    "metric": "prequential_mae",
                    "paired_seeds": int(len(merged)),
                    "mean_difference_sca_minus_baseline": float(difference.mean()),
                    "sca_wins": int(np.sum(difference < 0)),
                    "paired_t_pvalue": float(t_result.pvalue),
                    "wilcoxon_pvalue": w_pvalue,
                }
            )
    return seed_summary, method_summary, pd.DataFrame(tests)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--lag", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None, choices=(1, 12, 24))
    parser.add_argument("--split", choices=("legacy", "proceed"), default="legacy")
    parser.add_argument("--online-blocks", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.25)
    parser.add_argument("--feature-index", type=int, default=0)
    parser.add_argument("--base-epochs", type=int, default=20)
    parser.add_argument("--adapt-epochs", type=int, default=5)
    parser.add_argument("--null-windows", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--uncertain-alpha", type=float, default=0.10)
    parser.add_argument("--minimum-effect-z", type=float, default=2.5)
    parser.add_argument("--replay-size", type=int, default=512)
    parser.add_argument("--graph-top-k", type=int, default=8)
    parser.add_argument("--ewc-lambda", type=float, default=10.0)
    parser.add_argument("--der-alpha", type=float, default=0.5)
    parser.add_argument("--der-beta", type=float, default=0.5)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compact-time", action="store_true")
    parser.add_argument("--val-upgrade-tau", type=float, default=None)
    parser.add_argument("--revin", action="store_true")
    parser.add_argument("--weight-mod", action="store_true")
    parser.add_argument("--msg-adapter", action="store_true")
    parser.add_argument("--closed-form-time", action="store_true")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--gate-eta", type=float, default=1.0)
    parser.add_argument("--dual-view", choices=("off", "ftl", "eg", "eg_frozen"), default="off")
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--baseline-lr-scale", type=float, default=1.0)
    parser.add_argument("--mod-rank", type=int, default=2)
    parser.add_argument("--view-delta", type=float, default=0.03)
    parser.add_argument("--view-eta", type=float, default=1.0)
    parser.add_argument("--eval-batch", type=int, default=256,
                        help="Batch size of no-grad evaluation passes (does not change results).")
    parser.add_argument("--micro-batch", type=int, default=0,
                        help="Split each training batch into chunks of this size (gradient accumulation).")
    parser.add_argument("--route-variant", choices=ROUTE_VARIANTS, default="full",
                        help="Routing ablation variant (default reproduces the full router).")
    parser.add_argument("--no-memory", action="store_true",
                        help="Share one adapter state across actions (mechanism-memory ablation).")
    parser.add_argument("--warmup-validation", action="store_true",
                        help="Run on the warm-up segment only (hyperparameter selection).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global HORIZON, REPORT_STEPS
    if args.split == "proceed":
        HORIZON = 1 if args.horizon is None else args.horizon
        args.lag = 60 if args.lag is None else args.lag
    else:
        HORIZON = 12 if args.horizon is None else args.horizon
        args.lag = 12 if args.lag is None else args.lag
    REPORT_STEPS = reported_steps(HORIZON)
    ROUTE_OPTIONS["variant"] = str(args.route_variant)
    ROUTE_OPTIONS["no_memory"] = bool(args.no_memory)
    DV_OPTIONS["mode"] = args.dual_view
    TUNE_OPTIONS["lr_scale"] = float(args.lr_scale)
    TUNE_OPTIONS["baseline_lr_scale"] = float(args.baseline_lr_scale)
    TUNE_OPTIONS["mod_rank"] = int(args.mod_rank)
    DV_OPTIONS["delta"] = float(args.view_delta)
    DV_OPTIONS["eta"] = float(args.view_eta)
    MODEL_OPTIONS["compact_time"] = bool(args.compact_time)
    MODEL_OPTIONS["revin"] = bool(args.revin)
    MODEL_OPTIONS["weight_mod"] = bool(args.weight_mod)
    MODEL_OPTIONS["msg_adapter"] = bool(args.msg_adapter)
    FIT_OPTIONS["closed_form_time"] = bool(args.closed_form_time)
    FIT_OPTIONS["ridge"] = float(args.ridge)
    GATE_OPTIONS["enabled"] = bool(args.gate)
    GATE_OPTIONS["eta"] = float(args.gate_eta)
    UPGRADE_OPTIONS["tau"] = args.val_upgrade_tau
    MEMORY_OPTIONS["micro_batch"] = int(args.micro_batch)
    MEMORY_OPTIONS["eval_batch"] = int(args.eval_batch)
    if 1.0 / (args.null_windows + 1) > args.alpha:
        raise ValueError("null-windows is too small for the requested empirical alpha")
    if not args.smoke and (args.max_rows is not None or args.max_nodes is not None):
        raise ValueError("Subsetting is allowed only for a smoke test, never for formal evidence")
    selected_methods = tuple(args.methods)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    values, mean, std, data_metadata = load_stream_data(
        args.data,
        0.25 if args.split == "proceed" else args.warmup_ratio,
        args.feature_index,
        args.max_rows,
        args.max_nodes,
        args.warmup_validation,
    )
    dataset_name = args.dataset_name or args.data.stem
    digest = hashlib.sha256()
    with args.data.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    data_metadata["sha256"] = digest.hexdigest()
    protocol = {
        "evidence_type": "smoke_test" if args.smoke else ("warmup_validation" if args.warmup_validation else "newly_run"),
        "formal_configuration_frozen_before_run": not args.smoke,
        "dataset": dataset_name,
        "data": data_metadata,
        "protocol": {
            "warmup_online_ratio": (
                "0.20:0.05:0.75" if args.split == "proceed"
                else f"{args.warmup_ratio:.2f}:{1 - args.warmup_ratio:.2f}"
            ),
            "online_semantics": "predict block -> reveal labels -> diagnose -> update future block",
            "lookback": args.lag,
            "direct_output_horizon": HORIZON,
            "reported_steps": list(REPORT_STEPS),
            "reported_minutes": [15, 30, 60]
            if data_metadata["source_format"] == "npz"
            else None,
            "online_blocks": args.online_blocks,
            "normalization": data_metadata["normalization"],
            "primary_metrics": "warm-up-normalized prequential MSE and MAE",
            "secondary_metrics": (
                "masked original-scale MAE/RMSE/MAPE; targets equal to zero are excluded"
            ),
            "router": "empirical tail p-value AND robust standardized effect",
            "minimum_effect_z": args.minimum_effect_z,
            "compact_time": bool(args.compact_time),
            "revin": bool(args.revin),
            "weight_mod": bool(args.weight_mod),
            "msg_adapter": bool(args.msg_adapter),
            "closed_form_time": bool(args.closed_form_time),
            "ridge": args.ridge,
            "gate": bool(args.gate),
            "gate_eta": args.gate_eta,
            "lr_scale": args.lr_scale,
            "baseline_lr_scale": args.baseline_lr_scale,
            "mod_rank": args.mod_rank,
            "dual_view": args.dual_view,
            "view_delta": args.view_delta,
            "view_eta": args.view_eta,
            "val_upgrade_tau": args.val_upgrade_tau,
        },
        "methods": list(selected_methods),
        "method_set_policy": (
            "The exact PEMS08 formal method set is frozen and reused on every dataset."
        ),
        "hypotheses_and_gates": {
            "H1_accuracy_efficiency": (
                "DSGA is within 1% MAE of the best generic continual-learning baseline "
                "while updating less than 15% of its own parameters on average."
            ),
            "H2_routing_value": (
                "DSGA improves over the best fixed selective action in aggregate MAE; "
                "otherwise data-driven component routing is not supported on this stream."
            ),
            "H3_memory_value": (
                "DSGA improves over DSGA-NoMemory; otherwise action-keyed mechanism memory "
                "is not supported on this stream."
            ),
            "decision_rule": (
                "Claims are retained only when their stated numerical gate is met on the fixed run; "
                "no post-result retuning or rerun is permitted."
            ),
        },
        "training": {
            "seeds": args.seeds,
            "base_epochs_max": args.base_epochs,
            "adapt_epochs_max_per_revealed_block": args.adapt_epochs,
            "early_stopping_patience": 4,
            "replay_size": args.replay_size,
            "graph_top_k": args.graph_top_k,
            "ewc_lambda": args.ewc_lambda,
            "der_alpha": args.der_alpha,
            "der_beta": args.der_beta,
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "paper_protocol_anchors": [
            "TrafficStream (IJCAI 2021): L=12 and K=12 traffic forecasting",
            "OneNet (NeurIPS 2023): forecast backbones and generic CL baselines in one aligned online table",
            "DSOF (ICLR 2025): prediction of wholly unseen future horizons without information leakage and five seeds",
            "EAC (ICLR 2025): continual spatio-temporal graph forecasting with effectiveness and efficiency analyses",
            "DOL (IJCAI 2025): L=12, H=12 and representative ST/online forecasting baselines",
        ],
        "non_tuning_rule": "One fixed formal run; no result-dependent hyperparameter changes or reruns.",
    }
    (args.output / "preregistered_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    block_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    started = time.time()
    for seed in range(args.seeds):
        print(f"Starting seed={seed} on {device}", flush=True)
        seed_blocks, seed_routes, seed_history = run_seed(
            values,
            mean,
            std,
            int(data_metadata["warmup_end"]),
            seed,
            device,
            args.lag,
            args.online_blocks,
            args.base_epochs,
            args.adapt_epochs,
            args.null_windows,
            args.alpha,
            args.uncertain_alpha,
            args.minimum_effect_z,
            args.replay_size,
            args.graph_top_k,
            args.ewc_lambda,
            args.der_alpha,
            args.der_beta,
            selected_methods,
            args.split,
        )
        block_rows.extend(seed_blocks)
        route_rows.extend(seed_routes)
        history_rows.extend(seed_history)

    block_results = pd.DataFrame(block_rows)
    routing_results = pd.DataFrame(route_rows)
    final_history = pd.DataFrame(history_rows)
    seed_summary, method_summary, statistical_tests = summarize(block_results, final_history)
    block_results.to_csv(args.output / "block_results.csv", index=False)
    routing_results.to_csv(args.output / "routing_decisions.csv", index=False)
    final_history.to_csv(args.output / "final_history.csv", index=False)
    seed_summary.to_csv(args.output / "seed_summary.csv", index=False)
    method_summary.to_csv(args.output / "method_summary.csv", index=False)
    statistical_tests.to_csv(args.output / "statistical_tests.csv", index=False)

    target_counts = block_results.groupby(["seed", "method"])["normalized_count"].sum()
    if target_counts.groupby("seed").nunique().max() != 1:
        raise RuntimeError("Methods did not evaluate identical online targets")
    audit = {
        "primary_accuracy_comparison_aligned": True,
        "reason": (
            f"All methods predict identical complete {HORIZON}-step targets before their labels are revealed; "
            "normalization, stream order, block boundaries, and aggregation are shared."
        ),
        "seeds": args.seeds,
        "methods": len(selected_methods),
        "online_blocks_per_seed": args.online_blocks,
        "target_elements_per_seed_and_method": int(target_counts.iloc[0]),
        "future_target_leakage": False,
        "paper_reported_values_mixed_into_local_ranking": False,
    }
    (args.output / "protocol_alignment_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    protocol["training"]["elapsed_seconds"] = time.time() - started
    protocol["completion"] = "complete"
    (args.output / "metadata.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Completed {args.seeds} seed(s), {len(selected_methods)} methods; "
        f"artifacts={args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
