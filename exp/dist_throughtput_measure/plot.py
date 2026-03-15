#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def aggregate_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "dataset", "batch_size"], as_index=False)[metric]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    return grouped


def plot_family(df: pd.DataFrame, model: str, dataset: str, out_dir: str) -> None:
    subset = df[(df["model"] == model) & (df["dataset"] == dataset)].copy()
    if subset.empty:
        return

    subset = subset.sort_values(["mem_dim", "batch_size"])
    mem_dims = sorted(subset["mem_dim"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10.colors

    for i, mem_dim in enumerate(mem_dims):
        color = colors[i % len(colors)]
        sub = subset[subset["mem_dim"] == mem_dim]

        per_gpu_wall = aggregate_metric(sub, "per_gpu_wall_eps")
        x = per_gpu_wall["batch_size"].astype(int).to_numpy()

        ax.plot(x, per_gpu_wall["mean"], marker="o", linewidth=2.2, color=color,
                label=f"dim={mem_dim}")

    ax.set_title("Per-GPU Throughput (wall)")
    ax.set_xlabel("Batch Size per GPU (events)")
    ax.set_ylabel("Events / s / GPU")
    ax.set_xticks(sorted(subset["batch_size"].astype(int).unique()))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    plt.suptitle(f"{model} {dataset} distributed training throughput")
    plt.tight_layout()

    output_path = os.path.join(out_dir, f"{model}_{dataset}_throughput.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot distributed throughput curves from parsed CSV results.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input", type=str, default=os.path.join(script_dir, "results.csv"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(script_dir, "plots"))
    parser.add_argument("--model", action="append", default=None, help="optional model filter, repeatable")
    parser.add_argument("--dataset", action="append", default=None, help="optional dataset filter, repeatable")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    numeric_columns = [
        "batch_size",
        "mem_dim",
        "per_gpu_wall_eps",
        "per_gpu_gpu_stream_eps",
        "global_wall_eps",
        "global_gpu_stream_eps",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "batch_size",
            "mem_dim",
            "per_gpu_wall_eps",
            "per_gpu_gpu_stream_eps",
            "global_wall_eps",
            "global_gpu_stream_eps",
        ]
    ).copy()
    df["batch_size"] = df["batch_size"].astype(int)
    df["mem_dim"] = df["mem_dim"].astype(int)

    if args.model:
        df = df[df["model"].isin(args.model)]
    if args.dataset:
        df = df[df["dataset"].isin(args.dataset)]

    if df.empty:
        print("No rows available after filtering.")
        return

    for model, dataset in sorted(df[["model", "dataset"]].drop_duplicates().itertuples(index=False, name=None)):
        plot_family(df, model, dataset, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()