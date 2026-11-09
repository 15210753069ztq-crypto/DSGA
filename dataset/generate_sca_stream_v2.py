from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from e0_oracle.run_e0 import (
    dependency_shift,
    make_transition,
    simulate,
)


REGIMES = (
    ("stable_1", "base", "freeze"),
    ("marginal_1", "marginal", "time"),
    ("base_return_1", "base", "freeze"),
    ("dependency_1", "dependency", "graph"),
    ("base_return_2", "base", "freeze"),
    ("joint_1", "joint", "joint"),
    ("marginal_recurrence", "marginal", "time"),
    ("dependency_recurrence", "dependency", "graph"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empirical_checks(
    warmup: np.ndarray,
    segments: list[np.ndarray],
    mechanism_names: list[str],
    base_transition: np.ndarray,
    shifted_transition: np.ndarray,
) -> dict[str, object]:
    base_mean = warmup.mean(axis=0)
    base_covariance = np.cov(warmup, rowvar=False)
    base_covariance_norm = max(np.linalg.norm(base_covariance, ord="fro"), 1e-12)
    checks: dict[str, object] = {
        "transition_relative_change": float(
            np.linalg.norm(shifted_transition - base_transition, ord="fro")
            / max(np.linalg.norm(base_transition, ord="fro"), 1e-12)
        ),
        "regimes": [],
    }
    for segment, mechanism in zip(segments, mechanism_names, strict=True):
        checks["regimes"].append(
            {
                "mechanism": mechanism,
                "mean_shift_l2_per_node": float(
                    np.linalg.norm(segment.mean(axis=0) - base_mean)
                    / np.sqrt(segment.shape[1])
                ),
                "covariance_relative_change": float(
                    np.linalg.norm(
                        np.cov(segment, rowvar=False) - base_covariance,
                        ord="fro",
                    )
                    / base_covariance_norm
                ),
            }
        )
    return checks


def build_seed(
    seed: int,
    nodes: int,
    warmup_length: int,
    regime_length: int,
    strength: float,
) -> dict[str, object]:
    base_transition, base_support = make_transition(seed, nodes)
    shifted_transition, shifted_support = dependency_shift(
        base_transition,
        base_support,
        seed + 100_000,
        strength,
    )
    rng = np.random.default_rng(seed + 200_000)
    affected = rng.choice(nodes, size=max(2, round(0.30 * nodes)), replace=False)
    marginal_bias = np.zeros(nodes)
    marginal_bias[affected] = (
        rng.choice((-1.0, 1.0), size=len(affected)) * 0.22 * strength
    )
    marginal_scale = 1.0 + 0.55 * strength

    mechanisms = {
        "base": (base_transition, np.zeros(nodes), 1.0),
        "marginal": (base_transition, marginal_bias, marginal_scale),
        "dependency": (shifted_transition, np.zeros(nodes), 1.0),
        "joint": (shifted_transition, marginal_bias, marginal_scale),
    }
    warmup = simulate(
        *mechanisms["base"],
        warmup_length,
        np.random.default_rng(seed + 300_000),
    )
    segments = []
    mechanism_names = []
    for regime_index, (_, mechanism, _) in enumerate(REGIMES):
        transition, bias, scale = mechanisms[mechanism]
        segments.append(
            simulate(
                transition,
                bias,
                scale,
                regime_length,
                np.random.default_rng(seed + 400_000 + regime_index),
            )
        )
        mechanism_names.append(mechanism)

    values = np.concatenate((warmup, *segments), axis=0)
    starts = warmup_length + np.arange(len(REGIMES)) * regime_length
    ends = starts + regime_length
    checks = empirical_checks(
        warmup,
        segments,
        mechanism_names,
        base_transition,
        shifted_transition,
    )
    return {
        "values": values,
        "warmup_end": warmup_length,
        "regime_starts": starts,
        "regime_ends": ends,
        "regime_names": np.asarray([item[0] for item in REGIMES]),
        "mechanism_names": np.asarray(mechanism_names),
        "oracle_actions": np.asarray([item[2] for item in REGIMES]),
        "base_transition": base_transition,
        "shifted_transition": shifted_transition,
        "base_support": base_support,
        "shifted_support": shifted_support,
        "marginal_bias": marginal_bias,
        "marginal_scale": marginal_scale,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--warmup-length", type=int, default=4096)
    parser.add_argument("--regime-length", type=int, default=1536)
    parser.add_argument("--block-length", type=int, default=512)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    if args.regime_length % args.block_length:
        raise ValueError("regime-length must be divisible by block-length")
    online_length = len(REGIMES) * args.regime_length
    if args.warmup_length * 3 != online_length:
        raise ValueError("This benchmark fixes warm-up:online to 25:75")

    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    checks = []
    for seed in range(args.seeds):
        generated = build_seed(
            seed,
            args.nodes,
            args.warmup_length,
            args.regime_length,
            args.strength,
        )
        path = args.output / f"seed_{seed:02d}.npz"
        np.savez_compressed(
            path,
            values=generated["values"].astype(np.float32),
            warmup_end=np.asarray(generated["warmup_end"], dtype=np.int64),
            regime_starts=generated["regime_starts"].astype(np.int64),
            regime_ends=generated["regime_ends"].astype(np.int64),
            regime_names=generated["regime_names"],
            mechanism_names=generated["mechanism_names"],
            oracle_actions=generated["oracle_actions"],
            base_transition=generated["base_transition"].astype(np.float32),
            shifted_transition=generated["shifted_transition"].astype(np.float32),
            base_support=generated["base_support"].astype(np.int8),
            shifted_support=generated["shifted_support"].astype(np.int8),
            marginal_bias=generated["marginal_bias"].astype(np.float32),
            marginal_scale=np.asarray(generated["marginal_scale"], dtype=np.float32),
        )
        files.append(
            {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
        checks.append({"seed": seed, **generated["checks"]})

    manifest = {
        "name": "SCA-Stream-v2",
        "version": "2.0",
        "generator": "linear Gaussian graph VAR(1)",
        "purpose": "task-free continual forecasting and drift-mechanism diagnosis",
        "seeds": args.seeds,
        "nodes": args.nodes,
        "warmup_length": args.warmup_length,
        "online_length": online_length,
        "warmup_online_ratio": "25:75",
        "regime_length": args.regime_length,
        "block_length": args.block_length,
        "lag": 12,
        "horizon": 1,
        "regimes": [
            {"name": name, "mechanism": mechanism, "oracle_action": action}
            for name, mechanism, action in REGIMES
        ],
        "construction_guarantees": {
            "marginal_only": (
                "transition matrix is fixed; stationary mean and covariance change"
            ),
            "dependency_only": (
                "transition/support change; stationary mean is zero and stationary covariance "
                "is held at identity by Sigma = Gamma - A Gamma A^T"
            ),
            "joint": "both transition/support and stationary marginal change",
            "recurrence": "fresh samples are drawn from previously seen mechanisms",
            "task_labels": "stored only for evaluation and never exposed to the router",
        },
        "online_protocol": (
            "For every block: predict all targets before update, reveal labels, diagnose without "
            "task identity, then update for future blocks. Normalization uses warm-up only."
        ),
        "empirical_checks": checks,
        "files": files,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        "# SCA-Stream-v2\n\n"
        "A task-free prequential graph time-series stream for the original SCA idea. "
        "The hidden regime labels are evaluation metadata, not model inputs. The stream "
        "contains isolated marginal, dependency, joint, return, and recurrence events.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
