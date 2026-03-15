#!/usr/bin/env python3
"""Plot distributed timing benchmark results.

Generates two figures per (model, dataset, pin_memory):
1) line plot for forward/backward/allreduce timings
2) stacked area for forward breakdown (memory_updater, encoder, predictor)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_combo(df_combo: pd.DataFrame, model: str, dataset: str, pin_label: str, out_dir: str) -> None:
    df_combo = df_combo.sort_values("dim")
    dims = df_combo["dim"].astype(int).to_numpy()

    # Plot 1: forward/backward/allreduce
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, df_combo["forward_ms"], marker="o", linewidth=2.0, label="forward")
    ax.plot(dims, df_combo["backward_ms"], marker="s", linewidth=2.0, label="backward")
    ax.plot(dims, df_combo["bwd_to_first_ar_ms"], marker="^", linewidth=2.0, label="bwd->first allreduce")

    ax.set_title(f"{model} {dataset} {pin_label} rank timing")
    ax.set_xlabel("Memory Dim")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(dims)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()

    out1 = os.path.join(out_dir, f"{model}_{dataset}_{pin_label}_fb_ar.png")
    plt.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"  Saved {out1}")

    # Plot 2: forward breakdown
    mem = df_combo["memory_updater_ms"].to_numpy(dtype=float)
    enc = df_combo["encoder_ms"].to_numpy(dtype=float)
    pred = df_combo["predictor_ms"].to_numpy(dtype=float)

    cumulative = [np.zeros_like(mem), mem, mem + enc, mem + enc + pred]
    labels = ["memory_updater", "encoder", "predictor"]
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, color) in enumerate(zip(labels, colors)):
        lower = cumulative[i]
        upper = cumulative[i + 1]
        ax.fill_between(dims, lower, upper, alpha=0.75, color=color, label=label)
        ax.plot(dims, upper, color=color, linewidth=1.5, marker="o", markersize=4)

    ax.set_title(f"{model} {dataset} {pin_label} forward breakdown")
    ax.set_xlabel("Memory Dim")
    ax.set_ylabel("Time (ms)")
    ax.set_xticks(dims)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()

    out2 = os.path.join(out_dir, f"{model}_{dataset}_{pin_label}_forward_breakdown.png")
    plt.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"  Saved {out2}")


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input", type=str, default=os.path.join(script_dir, "results.csv"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(script_dir, "plots"))
    parser.add_argument("--rank", type=int, default=0, help="which rank rows to plot")
    parser.add_argument("--max_dim", type=int, default=512)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input)

    for c in ["dim", "rank"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["dim", "rank"]).copy()
    df["dim"] = df["dim"].astype(int)
    df["rank"] = df["rank"].astype(int)

    df = df[(df["rank"] == args.rank) & (df["dim"] <= args.max_dim)]

    if df.empty:
        print("No rows after filtering. Check --rank, --max_dim, and input data.")
        return

    keys = ["model", "dataset", "pin_memory"]
    for (model, dataset, pin_label), df_combo in df.groupby(keys):
        if len(df_combo) == 0:
            continue
        plot_combo(df_combo, model, dataset, pin_label, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
