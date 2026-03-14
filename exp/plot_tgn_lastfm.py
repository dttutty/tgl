#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timing_results.csv")
df_all = pd.read_csv(csv_path)
df_all["dim"] = pd.to_numeric(df_all["dim"], errors="coerce").astype("Int64")

STAGES = ["sample", "fetch_feature", "fetch_memory", "forward", "backward", "memory_update"]
COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948"]

MODELS   = ["TGN", "APAN", "JODIE"]
DATASETS = ["WIKI", "LASTFM", "MOOC", "REDDIT"]
PINS     = ["pin", "nopin"]

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(out_dir, exist_ok=True)


def plot_one(model, dataset, pin_label):
    df = df_all[
        (df_all["model"] == model) &
        (df_all["dataset"] == dataset) &
        (df_all["epoch"] == "avg") &
        (df_all["pin_memory"] == pin_label) &
        (df_all["dim"] <= 512)
    ].dropna(subset=["dim"]).sort_values("dim")

    if df.empty:
        print(f"  [skip] no data for {model}_{dataset}_{pin_label}")
        return

    dims = df["dim"].astype(int).values

    cumulative = [np.zeros(len(dims))]
    for stage in STAGES:
        cumulative.append(cumulative[-1] + df[stage].values)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (stage, color) in enumerate(zip(STAGES, COLORS)):
        lower = cumulative[i]
        upper = cumulative[i + 1]
        ax.fill_between(dims, lower, upper, alpha=0.75, color=color, label=stage)
        ax.plot(dims, upper, color=color, linewidth=1.5, marker="o", markersize=4)

    ax.plot(dims, cumulative[0], color="gray", linewidth=1, linestyle="--")

    ax.set_xlabel("Memory Dim")
    ax.set_ylabel("Cumulative Time (s)")
    ax.set_title(f"{model} · {dataset} ({pin_label}) — Avg Epoch Timing Breakdown")
    ax.set_xticks(dims)
    ax.legend(title="Stage", loc="upper left", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = f"{model}_{dataset}_{pin_label}.png"
    out = os.path.join(out_dir, fname)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


for model in MODELS:
    for dataset in DATASETS:
        for pin_label in PINS:
            plot_one(model, dataset, pin_label)

print(f"\nAll plots saved to {out_dir}/")
