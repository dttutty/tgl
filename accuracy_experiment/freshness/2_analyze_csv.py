#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import re

import pandas as pd
from scipy.stats import t


FRESHNESS_LOG_RE = re.compile(
    r"^(?:(?P<username>[^_]+)(?:_(?P<hostname>[^_]+))?_)?(?P<model>[^_]+)_(?P<dataset>[^_]+)_bs(?P<batch_size>\d+)_memdim(?P<dim_out>\d+)_ep(?P<epochs>\d+)_delay(?P<delay>\d+)_(?:run(?P<run_id>\d+)_)?pin\.log$"
)


def infer_username(row: pd.Series) -> str:
    if "username" in row and pd.notna(row["username"]) and str(row["username"]).strip():
        return str(row["username"]).strip()

    log_file = str(row.get("log_file", ""))
    name = Path(log_file).name
    m = FRESHNESS_LOG_RE.match(name)
    if m is None:
        return "unknown"

    username = m.group("username")
    return username if username else "unknown"


def ci95_halfwidth(var: float, n_runs: int) -> float:
    if pd.isna(var) or n_runs <= 1:
        return float("nan")
    se = math.sqrt(var / n_runs)
    return t.ppf(0.975, df=n_runs - 1) * se


def main():
    parser = argparse.ArgumentParser(
        description="统计 summary.tsv 中不同 username / batch_size / dim_out / delay 的 AP、AUC 均值、标准差、方差和 95% CI 半宽（按 run 聚合）"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/sqp17/Projects/frost/third_party/tgl/accuracy_experiment/freshness/logs/summary.tsv"),
        help="summary.tsv 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/sqp17/Projects/frost/third_party/tgl/accuracy_experiment/freshness/logs/summary_stats_by_dim_delay.tsv"),
        help="输出统计结果路径",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=None,
        help="期望每组应有的 run 数；默认自动使用当前文件中各组 n_runs 的最大值",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")

    # 转成数值，NA/非法值会变成 NaN
    df["test_ap"] = pd.to_numeric(df["test_ap"], errors="coerce")
    df["test_score"] = pd.to_numeric(df["test_score"], errors="coerce")
    df["username"] = df.apply(infer_username, axis=1)

    # 仅统计有有效指标的 run（通常 status=ok）
    valid = df.dropna(subset=["test_ap", "test_score"]).copy()

    # 以 username + batch_size + dim_out + delay 分组，统计 AP/AUC 的均值、标准差和方差（样本标准差/方差 ddof=1）
    stats = (
        valid.groupby(["username", "batch_size", "dim_out", "delay"], as_index=False)
        .agg(
            n_runs=("run_id", "count"),
            ap_mean=("test_ap", "mean"),
            ap_std=("test_ap", "std"),
            ap_var=("test_ap", "var"),
            auc_mean=("test_score", "mean"),
            auc_std=("test_score", "std"),
            auc_var=("test_score", "var"),
        )
        .sort_values(["username", "batch_size", "dim_out", "delay"])
    )

    stats["ap_ci95_halfwidth"] = stats.apply(
        lambda row: ci95_halfwidth(row["ap_var"], int(row["n_runs"])),
        axis=1,
    )
    stats["auc_ci95_halfwidth"] = stats.apply(
        lambda row: ci95_halfwidth(row["auc_var"], int(row["n_runs"])),
        axis=1,
    )

    expected_runs = args.expected_runs
    if expected_runs is None:
        expected_runs = int(stats["n_runs"].max()) if not stats.empty else 0

    stats["expected_runs"] = expected_runs

    # 保留 6 位小数，便于阅读
    for c in [
        "ap_mean",
        "ap_std",
        "ap_var",
        "ap_ci95_halfwidth",
        "auc_mean",
        "auc_std",
        "auc_var",
        "auc_ci95_halfwidth",
    ]:
        stats[c] = stats[c].round(6)

    # 标注该分组是否达到期望的 run 数
    stats["matches_expected_runs"] = stats["n_runs"] == stats["expected_runs"]

    # 输出到终端
    print(stats.to_string(index=False))

    # 保存结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.output, sep="\t", index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
