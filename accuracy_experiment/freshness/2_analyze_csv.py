#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="统计 summary.tsv 中不同 batch_size / dim_out / delay 的 AP、AUC 均值和方差（按 run 聚合）"
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
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")

    # 转成数值，NA/非法值会变成 NaN
    df["test_ap"] = pd.to_numeric(df["test_ap"], errors="coerce")
    df["test_score"] = pd.to_numeric(df["test_score"], errors="coerce")

    # 仅统计有有效指标的 run（通常 status=ok）
    valid = df.dropna(subset=["test_ap", "test_score"]).copy()

    # 以 batch_size + dim_out + delay 分组，统计 AP/AUC 的均值和方差（样本方差 ddof=1）
    stats = (
        valid.groupby(["batch_size", "dim_out", "delay"], as_index=False)
        .agg(
            n_runs=("run_id", "count"),
            ap_mean=("test_ap", "mean"),
            ap_var=("test_ap", "var"),
            auc_mean=("test_score", "mean"),
            auc_var=("test_score", "var"),
        )
        .sort_values(["batch_size", "dim_out", "delay"])
    )

    # 保留 6 位小数，便于阅读
    for c in ["ap_mean", "ap_var", "auc_mean", "auc_var"]:
        stats[c] = stats[c].round(6)

    # 标注是否满 10 个 run
    stats["is_full_10_runs"] = stats["n_runs"] == 10

    # 输出到终端
    print(stats.to_string(index=False))

    # 保存结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.output, sep="\t", index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
