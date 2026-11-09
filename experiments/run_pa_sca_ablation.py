"""协议 A 下的 SCA 受控流路由消融（5 条流 × 7 个变体 + z_min 敏感性）。

第 k 个在线块与 npz 中第 k 个 regime 对齐，因此块级 oracle 动作 = oracle_actions[k]。
    python run_pa_sca_ablation.py --max-gpus 2 --cpu-jobs 4 --threads 4
"""
from __future__ import annotations

import itertools

import pandas as pd

import run_pa_default as D
import run_pa_protocol as P

OUT = P.OUT / "sca_ablation2"
SEEDS = range(5)
VARIANTS = [("full", []), ("no_memory", ["--no-memory"]),
            ("no_effect_gate", ["--route-variant", "no_effect_gate"]),
            ("no_positivity_gate", ["--route-variant", "no_positivity_gate"]),
            ("marginal_only", ["--route-variant", "marginal_only"]),
            ("dependency_only", ["--route-variant", "dependency_only"]),
            ("always_joint", ["--route-variant", "always_joint"])]
Z_SENSITIVITY = (1.5, 2.5, 3.0, 3.5)


def sca_config(horizon: int) -> dict:
    best = pd.read_csv(P.OUT / "selection.csv")
    row = best[(best.dataset == "SCA") & (best.horizon == horizon)].iloc[0]
    return {k: row[k] for k in P.GRID}


def jobs(_datasets):
    seeds = list(SEEDS)
    out = []
    view = pd.read_csv(P.OUT / "mode_selection.csv")
    for s, h in itertools.product(seeds, P.HORIZONS):
        c = sca_config(h)
        data = P.DATA / f"sca_stream_v2_csv/sca_stream_v2_seed{s:02d}.csv"
        common = [P.MAIN[0], P.MAIN[1], "--data", str(data), "--dataset-name", "SCA",
                  "--horizon", str(h), *P.PROTOCOL[:-2]]
        m = view[(view.dataset == "SCA") & (view.horizon == h)]["mode"].iloc[0]
        view_mode = "eg" if m == "eg" else "ftl"
        runs = [(name, flags, c) for name, flags in VARIANTS]
        runs += [(f"z{z:g}", [], {**c, "minimum_effect_z": z}) for z in Z_SENSITIVITY
                 if float(z) != float(c["minimum_effect_z"])]
        for name, flags, cfg in runs:
            cfg_args = P.dv_args(cfg)
            i = cfg_args.index("--dual-view")
            cfg_args[i + 1] = view_mode
            target = OUT / name / f"seed{s:02d}_H{h}"
            cmd = common + ["--seeds", "1", "--methods", "DSGA", *cfg_args, *flags,
                            "--output", str(target)]
            out.append((f"sca_{name}_s{s}_H{h}", cmd, target, -P.SIZE["SCA"]))
    return out


if __name__ == "__main__":
    D.jobs, D.OUT = jobs, OUT
    D.main()
