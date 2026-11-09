from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


SCENARIOS = ("stable", "marginal", "dependency", "joint")
ACTIONS = ("freeze", "time", "graph", "joint", "full")
EXPECTED_ACTION = {
    "stable": "freeze",
    "marginal": "time",
    "dependency": "graph",
    "joint": "joint",
}


def make_transition(
    seed: int,
    n: int,
    density: float = 0.12,
    max_norm: float = 0.62,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    support = rng.random((n, n)) < density
    np.fill_diagonal(support, False)
    for target in range(n):
        if not support[target].any():
            candidates = [source for source in range(n) if source != target]
            support[target, int(rng.choice(candidates))] = True
    weights = rng.choice((-1.0, 1.0), size=(n, n)) * rng.uniform(0.07, 0.18, size=(n, n))
    transition = support * weights
    np.fill_diagonal(transition, rng.uniform(0.18, 0.32, size=n))
    off_diagonal = ~np.eye(n, dtype=bool)
    for _ in range(100):
        if np.linalg.norm(transition, 2) <= max_norm:
            break
        transition[off_diagonal] *= 0.95
    if np.linalg.norm(transition, 2) > max_norm:
        raise RuntimeError("Could not stabilize the base transition")
    return transition, support


def dependency_shift(
    base: np.ndarray,
    support: np.ndarray,
    seed: int,
    strength: float,
    max_norm: float = 0.62,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shifted = base.copy()
    shifted_support = support.copy()
    existing = np.argwhere(support)
    possible = np.argwhere(~support & ~np.eye(len(base), dtype=bool))
    changed_fraction = min(0.55, 0.18 + 0.17 * strength)
    count = max(2, round(len(existing) * changed_fraction))
    removed = existing[rng.choice(len(existing), count, replace=False)]
    added = possible[rng.choice(len(possible), count, replace=False)]
    for target, source in removed:
        shifted_support[target, source] = False
        shifted[target, source] = 0.0
    for target, source in added:
        shifted_support[target, source] = True
        shifted[target, source] = rng.choice((-1.0, 1.0)) * rng.uniform(
            0.08 + 0.03 * strength,
            0.15 + 0.05 * strength,
        )
    remaining = np.argwhere(shifted_support)
    weight_count = max(2, round(len(remaining) * changed_fraction))
    selected = remaining[rng.choice(len(remaining), weight_count, replace=False)]
    for target, source in selected:
        shifted[target, source] += rng.choice((-1.0, 1.0)) * rng.uniform(
            0.04 * strength,
            0.10 * strength,
        )

    off_diagonal = ~np.eye(len(base), dtype=bool)
    for _ in range(100):
        if np.linalg.norm(shifted, 2) <= max_norm:
            break
        shifted[off_diagonal] *= 0.95
    if np.linalg.norm(shifted, 2) > max_norm:
        raise RuntimeError("Could not stabilize dependency shift")
    return shifted, shifted_support


def innovation_covariance(transition: np.ndarray, marginal_scale: float) -> np.ndarray:
    gamma = marginal_scale * np.eye(len(transition))
    covariance = gamma - transition @ gamma @ transition.T
    covariance = (covariance + covariance.T) / 2.0
    if np.linalg.eigvalsh(covariance).min() <= 1e-8:
        raise RuntimeError("Innovation covariance is not positive definite")
    return covariance


def simulate(
    transition: np.ndarray,
    bias: np.ndarray,
    marginal_scale: float,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    covariance = innovation_covariance(transition, marginal_scale)
    stationary_mean = np.linalg.solve(np.eye(len(transition)) - transition, bias)
    x = np.empty((length, len(transition)), dtype=np.float64)
    x[0] = rng.multivariate_normal(stationary_mean, marginal_scale * np.eye(len(transition)))
    noise = rng.multivariate_normal(np.zeros(len(transition)), covariance, size=length - 1)
    for t in range(1, length):
        x[t] = bias + transition @ x[t - 1] + noise[t - 1]
    return x


def generate_suite(
    seed: int,
    strength: float,
    n: int = 20,
    length: int = 10_000,
) -> dict[str, object]:
    base_a, base_support = make_transition(seed, n)
    shifted_a, shifted_support = dependency_shift(
        base_a,
        base_support,
        seed + 100_000,
        strength,
    )
    rng = np.random.default_rng(seed + 200_000)
    affected = rng.choice(n, size=max(2, round(0.30 * n)), replace=False)
    marginal_bias = np.zeros(n)
    marginal_bias[affected] = rng.choice((-1.0, 1.0), size=len(affected)) * 0.22 * strength
    marginal_scale = 1.0 + 0.55 * strength

    base = simulate(base_a, np.zeros(n), 1.0, length, np.random.default_rng(seed + 300_000))
    streams = {
        "stable": simulate(
            base_a, np.zeros(n), 1.0, length, np.random.default_rng(seed + 400_000)
        ),
        "marginal": simulate(
            base_a, marginal_bias, marginal_scale, length, np.random.default_rng(seed + 500_000)
        ),
        "dependency": simulate(
            shifted_a, np.zeros(n), 1.0, length, np.random.default_rng(seed + 600_000)
        ),
        "joint": simulate(
            shifted_a,
            marginal_bias,
            marginal_scale,
            length,
            np.random.default_rng(seed + 700_000),
        ),
    }
    return {
        "base_a": base_a,
        "shifted_a": shifted_a,
        "base_support": base_support,
        "shifted_support": shifted_support,
        "marginal_bias": marginal_bias,
        "marginal_scale": marginal_scale,
        "base": base,
        "streams": streams,
    }


def pairs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = values[:-1].T
    x_augmented = np.vstack((x, np.ones(x.shape[1])))
    return x_augmented, values[1:].T


def ridge_estimate(x: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    penalty = ridge * np.eye(x.shape[0])
    penalty[-1, -1] = ridge * 0.1
    return (y @ x.T) @ np.linalg.inv(x @ x.T + penalty)


def action_masks(n: int, base_support: np.ndarray) -> dict[str, np.ndarray]:
    diagonal = np.eye(n, dtype=bool)
    time_mask = np.zeros((n, n + 1), dtype=bool)
    time_mask[:, :n] = diagonal
    time_mask[:, -1] = True
    graph_mask = np.zeros((n, n + 1), dtype=bool)
    graph_mask[:, :n] = base_support
    full_mask = np.zeros((n, n + 1), dtype=bool)
    full_mask[:, :n] = np.ones((n, n), dtype=bool)
    full_mask[:, -1] = True
    return {
        "freeze": np.zeros((n, n + 1), dtype=bool),
        "time": time_mask,
        "graph": graph_mask,
        "joint": time_mask | graph_mask,
        "full": full_mask,
    }


def masked_proximal_update(
    x: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    mask: np.ndarray,
    proximal: float,
) -> np.ndarray:
    updated = base.copy()
    for target in range(len(base)):
        active = np.flatnonzero(mask[target])
        if len(active) == 0:
            continue
        frozen = ~mask[target]
        target_residual = y[target] - base[target, frozen] @ x[frozen]
        active_x = x[active]
        center = base[target, active]
        gram = active_x @ active_x.T + proximal * np.eye(len(active))
        rhs = active_x @ target_residual + proximal * center
        updated[target, active] = np.linalg.solve(gram, rhs)
    return updated


def mse(parameters: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((parameters @ x - y) ** 2))


def parameter_count(mask: np.ndarray) -> int:
    return int(mask.sum())


def run_experiment(
    seeds: range,
    strengths: list[float],
    sample_sizes: list[int],
    proximal: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for strength in strengths:
            generated = generate_suite(seed, strength)
            base_x, base_y = pairs(generated["base"][:7000])
            base_estimate = ridge_estimate(base_x, base_y)
            history_x, history_y = pairs(generated["base"][7000:])
            base_history_mse = mse(base_estimate, history_x, history_y)
            masks = action_masks(len(generated["base_a"]), generated["base_support"])
            full_parameter_count = parameter_count(masks["full"])

            for scenario in SCENARIOS:
                stream = generated["streams"][scenario]
                current_x, current_y = pairs(stream[5000:])
                for sample_size in sample_sizes:
                    adapt_x, adapt_y = pairs(stream[: sample_size + 1])
                    for action in ACTIONS:
                        if action == "freeze":
                            updated = base_estimate.copy()
                        elif action == "full":
                            updated = ridge_estimate(adapt_x, adapt_y, ridge=1.0)
                        else:
                            updated = masked_proximal_update(
                                adapt_x,
                                adapt_y,
                                base_estimate,
                                masks[action],
                                proximal,
                            )
                        current_mse = mse(updated, current_x, current_y)
                        history_mse = mse(updated, history_x, history_y)
                        trainable = parameter_count(masks[action])
                        forgetting = max(history_mse - base_history_mse, 0.0)
                        utility = (
                            current_mse
                            + 0.25 * forgetting
                            + 0.002 * trainable / max(full_parameter_count, 1)
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "strength": strength,
                                "scenario": scenario,
                                "sample_size": sample_size,
                                "action": action,
                                "expected_action": EXPECTED_ACTION[scenario],
                                "current_mse": current_mse,
                                "history_mse": history_mse,
                                "base_history_mse": base_history_mse,
                                "forgetting": forgetting,
                                "utility": utility,
                                "trainable_parameters": trainable,
                            }
                        )
    frame = pd.DataFrame(rows)
    frame["current_rank"] = frame.groupby(
        ["seed", "strength", "scenario", "sample_size"]
    )["current_mse"].rank(method="average")
    frame["utility_rank"] = frame.groupby(
        ["seed", "strength", "scenario", "sample_size"]
    )["utility"].rank(method="average")
    return frame


def paired_direction(
    frame: pd.DataFrame,
    scenario: str,
    preferred: str,
    comparator: str,
) -> dict[str, object]:
    selected = frame[frame.scenario == scenario]
    pivot = selected.pivot(
        index=["seed", "strength", "sample_size"],
        columns="action",
        values="utility",
    )
    delta = pivot[preferred] - pivot[comparator]
    return {
        "scenario": scenario,
        "preferred": preferred,
        "comparator": comparator,
        "preferred_minus_comparator": float(delta.mean()),
        "preferred_win_fraction": float((delta < 0).mean()),
        "paired_p": float(ttest_rel(pivot[preferred], pivot[comparator]).pvalue),
        "comparisons": int(len(delta)),
    }


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    aggregate = (
        frame.groupby(["strength", "scenario", "sample_size", "action"])
        .agg(
            current_mse=("current_mse", "mean"),
            history_mse=("history_mse", "mean"),
            forgetting=("forgetting", "mean"),
            utility=("utility", "mean"),
            current_rank=("current_rank", "mean"),
            utility_rank=("utility_rank", "mean"),
        )
        .reset_index()
    )
    directions = [
        paired_direction(frame, "marginal", "time", "graph"),
        paired_direction(frame, "dependency", "graph", "time"),
        paired_direction(frame, "joint", "joint", "time"),
        paired_direction(frame, "joint", "joint", "graph"),
    ]

    stable = frame[frame.scenario == "stable"]
    stable_pivot = stable.pivot(
        index=["seed", "strength", "sample_size"],
        columns="action",
        values="utility",
    )
    stable_competitors = stable_pivot[["time", "graph", "joint", "full"]].min(axis=1)
    stable_delta = stable_pivot["freeze"] - stable_competitors
    stable_direction = {
        "scenario": "stable",
        "preferred": "freeze",
        "comparator": "best_update",
        "preferred_minus_comparator": float(stable_delta.mean()),
        "preferred_win_fraction": float((stable_delta < 0).mean()),
        "paired_p": float(ttest_rel(stable_pivot["freeze"], stable_competitors).pvalue),
        "comparisons": int(len(stable_delta)),
    }
    directions.insert(0, stable_direction)

    expected = frame[frame.action == frame.expected_action]
    expected_top2 = float((expected.utility_rank <= 2).mean())
    individual_passes = {
        "stable": directions[0]["preferred_win_fraction"] >= 0.60,
        "marginal": directions[1]["preferred_win_fraction"] >= 0.60,
        "dependency": directions[2]["preferred_win_fraction"] >= 0.60,
        "joint": (
            directions[3]["preferred_win_fraction"] >= 0.60
            and directions[4]["preferred_win_fraction"] >= 0.60
        ),
    }
    hypothesis_passes = sum(individual_passes.values())
    if hypothesis_passes == 4 and expected_top2 >= 0.60:
        support_level = "strong_support"
    elif hypothesis_passes >= 2:
        support_level = "trend_support"
    else:
        support_level = "not_supported"
    assessment = {
        "support_level": support_level,
        "hypothesis_passes_at_60pct": hypothesis_passes,
        "individual_hypothesis_passes": individual_passes,
        "expected_action_top2_fraction": expected_top2,
        "directional_tests": directions,
        "utility_definition": "current_mse + 0.25*positive_forgetting + 0.002*trainable_fraction",
        "interpretation": (
            "E0 tests component specialization with Oracle drift labels. "
            "It does not test automatic drift diagnosis."
        ),
    }
    return aggregate, assessment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--proximal", type=float, default=10.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Running E0 Oracle component-specialization experiment", flush=True)
    results = run_experiment(
        range(args.seeds),
        args.strengths,
        args.sample_sizes,
        args.proximal,
    )
    results.to_csv(args.output / "e0_results.csv", index=False)
    aggregate, assessment = summarize(results)
    aggregate.to_csv(args.output / "e0_aggregate.csv", index=False)
    (args.output / "e0_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "strengths": args.strengths,
                "sample_sizes": args.sample_sizes,
                "proximal": args.proximal,
                "actions": ACTIONS,
                "scenarios": SCENARIOS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(assessment, indent=2), flush=True)


if __name__ == "__main__":
    main()
