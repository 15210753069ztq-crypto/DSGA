"""协议 A 下双视图消融：用每个数据集/步长选出的配置，只改变扩展组件。

变体：
  mod       仅低秩权重调制（不开双视图）
  ens       冻结的 A+B 视图集成（不调用适配器、不开调制）
  ens_full  A+B 集成 + 适配器 + 调制
完整 DSGA（FTL 切换）复用 outputs/pa2/final 与 final_fix。
    python run_pa_dvablation.py --max-gpus 2 --cpu-jobs 2 --threads 4
"""
from __future__ import annotations

import itertools

import pandas as pd

import run_pa_default as D
import run_pa_protocol as P

OUT = P.OUT / "dv_ablation"


def variant_args(c: dict, variant: str) -> list[str]:
    args = P.dv_args(c)
    i = args.index("--dual-view")
    if variant == "mod":
        args[i + 1] = "off"
    elif variant == "ens":
        args[i + 1] = "eg_frozen"
        args.remove("--weight-mod")
    elif variant == "ens_full":
        args[i + 1] = "eg"
    return args


def jobs(datasets):
    best = pd.read_csv(P.OUT / "selection.csv")
    out = []
    for d, h, v in itertools.product(datasets, P.HORIZONS, ("mod", "ens", "ens_full")):
        row = best[(best.dataset == d) & (best.horizon == h)].iloc[0]
        c = {k: row[k] for k in P.GRID}
        target = OUT / v / f"{d}_H{h}"
        cmd = P.MAIN + P.data_args(d, h) + P.PROTOCOL[:-2] + [
            "--seeds", "5", "--methods", "Frozen", "DSGA", *variant_args(c, v), "--output", str(target)]
        out.append((f"abl_{v}_{d}_H{h}", cmd, target, -P.SIZE[d]))
    return sorted(out, key=lambda j: j[3])


if __name__ == "__main__":
    D.jobs, D.OUT = jobs, OUT
    D.main()
