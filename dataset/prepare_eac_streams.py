#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EAC (ICLR'25) Air-Stream / Energy-Stream 数据获取与整理脚本。

用法:
  python prepare_eac_streams.py download [air|energy|all]   # 从官方 SharePoint 匿名下载 RawData + graph
  python prepare_eac_streams.py build    [air|energy|all]   # 整理为固定节点连续流 CSV + README

下载方式: 先 GET 分享链接拿到匿名 FedAuth cookie, 再用 SharePoint REST
  /_api/web/GetFileByServerRelativeUrl('<path>')/$value
下载, 支持 Range 断点续传与重试 (网络不稳定)。
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
SHARE_URL = ("https://hkustgz-my.sharepoint.com/:f:/g/personal/wchen110_connect_hkust-gz_edu_cn/"
             "EuiKtt95qnpNgOngXAV_MmABWYyEBh74ooM94kdycwg4Sw?e=ZRCC1n")
SP_BASE = "https://hkustgz-my.sharepoint.com/personal/wchen110_connect_hkust-gz_edu_cn"
SP_ROOT = "/personal/wchen110_connect_hkust-gz_edu_cn/Documents/data"

DATASETS = {
    "air": {"remote": "AIR", "out": "air_stream", "periods": ["2016", "2017", "2018", "2019"]},
    "energy": {"remote": "ENERGY-Wind", "out": "energy_stream", "periods": ["0", "1", "2", "3"]},
}
CHUNK = 4 << 20  # 每次 Range 请求 4MB


def new_session():
    s = requests.Session()
    s.get(SHARE_URL, timeout=60)  # 获取匿名 FedAuth cookie
    return s


def remote_size(s, rel):
    r = s.get(f"{SP_BASE}/_api/web/GetFileByServerRelativeUrl('{rel}')",
              headers={"Accept": "application/json;odata=nometadata"}, timeout=60)
    r.raise_for_status()
    return int(r.json()["Length"])


def download_file(rel, dst, max_retry=200):
    """分块 Range 下载, 断点续传; 返回最终文件大小。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    s = new_session()
    total = remote_size(s, rel)
    part = dst + ".part"
    if os.path.exists(dst) and os.path.getsize(dst) == total:
        print(f"[skip] {dst} ({total} B)", flush=True)
        return total
    done = os.path.getsize(part) if os.path.exists(part) else 0
    url = f"{SP_BASE}/_api/web/GetFileByServerRelativeUrl('{rel}')/$value"
    retry = 0
    while done < total:
        end = min(done + CHUNK, total) - 1
        try:
            r = s.get(url, headers={"Range": f"bytes={done}-{end}"}, timeout=120)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            if r.status_code == 200 and done > 0:
                raise RuntimeError("服务器不支持 Range")
            data = r.content
            with open(part, "ab") as f:
                f.write(data)
            done += len(data)
            retry = 0
            print(f"  {os.path.basename(dst)} {done}/{total} ({100*done/total:.1f}%)", flush=True)
        except Exception as e:  # 网络错误: 重建会话后重试
            retry += 1
            if retry > max_retry:
                raise
            print(f"  [retry {retry}] {e}", flush=True)
            time.sleep(min(30, 2 * retry))
            try:
                s = new_session()
            except Exception:
                pass
    os.replace(part, dst)
    print(f"[ok] {dst} ({total} B)", flush=True)
    return total


def download(key):
    cfg = DATASETS[key]
    for sub, suf in [("graph", "_adj.npz"), ("RawData", ".npz")]:
        for p in cfg["periods"]:
            rel = f"{SP_ROOT}/{cfg['remote']}/{sub}/{p}{suf}"
            dst = f"{ROOT}/{cfg['out']}/raw/{sub}/{p}{suf}"
            download_file(rel, dst)
# ---------------------------------------------------------------------------
# 整理: 固定节点连续流
# EAC/TrafficStream 约定: 第 p+1 期节点 = 第 p 期节点(同序, 位于前缀) + 尾部新增节点。
# 因此所有时期都存在的节点交集 = 前 min(N_p) 列 (= 第一期全部节点)。
# 对该约定做两项校验并写入 README:
#   (1) graph/{p}_adj.npz 是否等于 graph/{p+1}_adj.npz 左上 N_p x N_p 子矩阵;
#   (2) 期间边界处同索引节点的跳变是否远小于随机置换后的跳变。
# ---------------------------------------------------------------------------
META = {
    "air": {"feature": "AQI (空气质量指数; EAC 论文: predict future air quality index (AQI) flow)",
            "freq": "1 小时", "period_desc": "每期 = 一个自然年 (2016/2017/2018/2019)",
            "source": "quotsoft.net/air 全国空气质量监测站数据, 经 EAC 作者处理",
            "notes": ["名义采样间隔 1 小时, 但每年行数 (2016: 8578, 2017: 8619, 2018: 8378, 2019: 8490) 少于该年小时数"
                      " (8784/8760/8760/8760), 说明 EAC 处理时删去了部分时刻; 行之间不严格等间隔, 且无时间戳无法还原。"
                      " 总帧数 34065 与 EAC 论文 Table 一致。",
                      "原始 AQI 取值 0~500 (国标上限截断); 边界处同索引跳变约为随机置换的 0.4~0.55,"
                      " 主要依据 adj 前缀完全一致判断节点顺序一致。"]},
    "energy": {"feature": ("风机机舱内温度 (SDWPF 字段 Itmp; EAC 论文原文: using temperature inside the "
                           "turbine nacelle as a substitute for active power flow observations), 不是 Patv"),
               "freq": "10 分钟", "period_desc": "4 期, 编号 0/1/2/3",
               "source": "百度 KDD Cup 2022 SDWPF 风电数据 (134 台风机, 245 天), 经 EAC 作者处理",
               "notes": ["每期 8640 步 = 60 天, 4 期共 240 天 (34560 帧, 与 EAC 论文 Table 一致), SDWPF 原始 245 天中约 5 天未使用。",
                         "EAC 的 conf/ENERGY-Wind/*.json 未指定特征名; 特征依据 EAC 论文文字 (机舱内温度) 与数值范围 (-22.56~70.93) 判断,"
                         " 不是 Patv。列号为 EAC 节点索引, 与 SDWPF TurbID 的对应关系 EAC 未提供。"]},
}


def ordering_checks(X, A):
    rows = []
    rng = np.random.default_rng(0)
    for p in range(len(X) - 1):
        n = A[p].shape[0]
        adj_ok = bool(np.allclose(A[p], A[p + 1][:n, :n]))
        a, b = X[p][-1, :n], X[p + 1][0, :n]
        same = float(np.nanmean(np.abs(a - b)))
        perm = float(np.mean([np.nanmean(np.abs(a - b[rng.permutation(n)])) for _ in range(50)]))
        rows.append((p, p + 1, adj_ok, same, perm))
    return rows


def build(key):
    cfg = DATASETS[key]
    out_dir = f"{ROOT}/{cfg['out']}"
    X = [np.load(f"{out_dir}/raw/RawData/{p}.npz")["x"].astype(np.float64) for p in cfg["periods"]]
    A = [np.load(f"{out_dir}/raw/graph/{p}_adj.npz")["x"] for p in cfg["periods"]]
    for x, a in zip(X, A):
        assert x.shape[1] == a.shape[0], "数据列数与图节点数不一致"
    sizes = [x.shape[1] for x in X]
    assert sizes == sorted(sizes), "节点数不是单调扩张, 前缀交集约定不成立"
    n_common = min(sizes)
    checks = ordering_checks(X, A)

    data = np.concatenate([x[:, :n_common] for x in X], axis=0)
    bounds, s = [], 0
    for p, x in zip(cfg["periods"], X):
        bounds.append((p, s, s + x.shape[0] - 1, x.shape[0]))
        s += x.shape[0]
    raw_nan = np.isnan(data)
    stats = {
        "n_nan": int(raw_nan.sum()),
        "cover_total": 1 - raw_nan.mean(),
        "cover_period": [1 - np.isnan(x[:, :n_common]).mean() for x in X],
        "cover_period_full": [1 - np.isnan(x).mean() for x in X],
        "zero_ratio": float((data == 0).mean()),
        "zero_period_full": [float((x == 0).mean()) for x in X],
        "node_cover": 1 - raw_nan.mean(axis=0),
    }
    df = pd.DataFrame(data, columns=[str(i) for i in range(n_common)])
    df = df.ffill().bfill()  # 缺失值: 前向填充再后向填充
    assert not df.isna().any().any()
    csv_path = f"{out_dir}/{cfg['out']}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g")
    print(f"[build] {csv_path} shape={df.shape}", flush=True)
    write_readme(key, out_dir, csv_path, df.shape, sizes, n_common, bounds, checks, stats)


def write_readme(key, out_dir, csv_path, shape, sizes, n_common, bounds, checks, st):
    cfg, meta = DATASETS[key], META[key]
    P = cfg["periods"]
    L = [f"# {cfg['out']} (EAC ICLR'25 {cfg['remote']} 数据, 固定节点连续流)\n", "## 来源"]
    L.append(f"- 原始来源: {meta['source']}")
    L.append(f"- 官方处理后数据: EAC 作者 SharePoint 分享 {SHARE_URL}")
    L.append(f"  - 远程 `data/{cfg['remote']}/RawData/{{p}}.npz` (键 `x`, 形状 (T_p, N_p)) 与 "
             f"`data/{cfg['remote']}/graph/{{p}}_adj.npz` (键 `x`, 形状 (N_p, N_p))")
    L.append(f"  - 本地 `{out_dir}/raw/RawData/` 与 `{out_dir}/raw/graph/`"
             "; FastData (训练样本切片, 可由 RawData 重建, 体积大) 未下载")
    L.append("- 生成脚本: `dataset/prepare_eac_streams.py`"
             " (`python prepare_eac_streams.py download|build air|energy|all`)\n")
    L.append("## 输出")
    L.append(f"- `{csv_path}`: 形状 (T, N) = ({shape[0]}, {shape[1]}), 只含数值列, 列名为 EAC 节点索引 `0..{n_common-1}`")
    L.append(f"- 特征: {meta['feature']}")
    L.append(f"- 采样间隔: {meta['freq']}; {meta['period_desc']}")
    L.append("- 数值为 EAC RawData 原值, 未标准化; 时期之间首尾直接拼接; EAC 未提供绝对时间戳, CSV 无时间列\n")
    L.append("## 节点")
    L.append("- 各时期节点数: " + ", ".join(f"{p}: {n}" for p, n in zip(P, sizes)))
    L.append(f"- 所有时期交集节点数: {n_common}. EAC 约定新节点追加在尾部 (main.py: `range(pre_node_size, cur_node_size)`),"
             f" 故交集 = 每期前 {n_common} 列")
    L.append("- 前缀约定校验:")
    for p, q, ok, same, perm in checks:
        L.append(f"  - {P[p]} -> {P[q]}: adj 前缀子矩阵一致={ok}; 边界处同索引平均跳变 {same:.4f}, 随机置换 {perm:.4f}")
    L.append("\n## 时期边界 (CSV 数据行号, 0-based, 不含表头, 闭区间)\n")
    L.append("| 时期 | 起始行 | 结束行 | 长度 |")
    L.append("|---|---|---|---|")
    for p, s, e, n in bounds:
        L.append(f"| {p} | {s} | {e} | {n} |")
    L.append("\n## 覆盖率 (填充前非 NaN 比例)")
    L.append(f"- 交集节点整体: {st['cover_total']:.6f} (NaN 共 {st['n_nan']} 个, 已前向填充再后向填充)")
    L.append("- 各时期 (交集节点 / 该期全部节点): " + ", ".join(
        f"{p}: {a:.6f} / {b:.6f}" for p, a, b in zip(P, st["cover_period"], st["cover_period_full"])))
    L.append(f"- 单节点覆盖率 min / median: {st['node_cover'].min():.6f} / {np.median(st['node_cover']):.6f}")
    L.append(f"- 交集数据中精确为 0 的比例: {st['zero_ratio']:.6f}; 各期全部节点: " + ", ".join(
        f"{p}: {z:.6f}" for p, z in zip(P, st["zero_period_full"])) + " (0 值保留原样, 未视为缺失)")
    L.append("\n## 偏差与说明")
    L.extend(f"- {n}" for n in meta.get("notes", []))
    with open(f"{out_dir}/README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[build] {out_dir}/README.md", flush=True)



if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    keys = ["energy", "air"] if which == "all" else [which]
    for k in keys:
        if cmd in ("download", "all"):
            download(k)
        if cmd in ("build", "all"):
            build(k)
