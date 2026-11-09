"""协议 A：在验证流上为每个数据集/步长选择视图组合方式（ftl 切换 或 eg 加权集成）。

沿用 selection.csv 中已选的超参数，只为 eg 模式补跑验证流；ftl 的验证 MSE 来自原搜索。
    python run_pa_modesel.py --max-gpus 2 --cpu-jobs 4 --threads 4   # 跑验证
    python run_pa_modesel.py --select                                 # 汇总选择
"""
from __future__ import annotations

import itertools
import sys

import pandas as pd

import run_pa_default as D
import run_pa_protocol as P
from run_pa_dvablation import variant_args

OUT = P.OUT / "search_eg"


def jobs(datasets):
    best = pd.read_csv(P.OUT / "selection.csv")
    out = []
    for d, h in itertools.product(datasets, P.HORIZONS):
        row = best[(best.dataset == d) & (best.horizon == h)].iloc[0]
        c = {k: row[k] for k in P.GRID}
        target = OUT / f"{d}_H{h}"
        cmd = P.MAIN + P.data_args(d, h) + P.PROTOCOL[:-2] + [
            "--seeds", str(P.SEARCH_SEEDS.get(d, 3)), "--methods", "DSGA", "--warmup-validation",
            *variant_args(c, "ens_full"), "--output", str(target)]
        out.append((f"modesel_{d}_H{h}", cmd, target, -P.SIZE[d]))
    return sorted(out, key=lambda j: j[3])


def select() -> pd.DataFrame:
    best = pd.read_csv(P.OUT / "selection.csv")
    rows = []
    for r in best.itertuples():
        f = pd.read_csv(OUT / f"{r.dataset}_H{r.horizon}" / "method_summary.csv")
        eg = float(f[f.method == "DSGA"].prequential_mse_mean.iloc[0])
        rows.append({"dataset": r.dataset, "horizon": r.horizon, "val_ftl": r.val_mse, "val_eg": eg,
                     "mode": "eg" if eg < r.val_mse else "ftl"})
    res = pd.DataFrame(rows)
    res.to_csv(P.OUT / "mode_selection.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    return res


if __name__ == "__main__":
    if "--select" in sys.argv:
        select()
    else:
        D.jobs, D.OUT = jobs, OUT
        D.main()
