#!/usr/bin/env python3
"""
One horizontal stacked-bar figure per CSV row.
Blocks (left to right):
  1. memory_updater_ms
  2. encoder_ms
  3. predictor_ms
  4. bwd_to_first_ar_ms
  5. backward_ms - bwd_to_first_ar_ms  (remaining backward after first AR)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SEGMENTS = [
    ("memory_updater",  "#4e79a7"),
    ("encoder",         "#f28e2b"),
    ("predictor",       "#59a14f"),
    ("bwd→1st AR",      "#e15759"),
    ("bwd remainder",   "#76b7b2"),
]
COLORS = [c for _, c in SEGMENTS]
LABELS = [l for l, _ in SEGMENTS]

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "results_lastfm_rank0.csv")
out_dir    = os.path.join(script_dir, "plots_per_row")
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(input_path)

for _, row in df.iterrows():
    values = [
        row["memory_updater_ms"],
        row["encoder_ms"],
        row["predictor_ms"],
        row["bwd_to_first_ar_ms"],
        max(0.0, row["backward_ms"] - row["bwd_to_first_ar_ms"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 1.4))

    left = 0.0
    for val, color in zip(values, COLORS):
        ax.barh(0, val, left=left, height=0.5, color=color)
        if val > 0.3:
            ax.text(left + val / 2, 0, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        left += val

    title = f"{row['model']} | {row['dataset']} | dim={row['dim']}"
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("ms")
    ax.set_yticks([])
    ax.set_xlim(0, left * 1.02)

    patches = [mpatches.Patch(color=c, label=l) for l, c in SEGMENTS]
    ax.legend(handles=patches, loc="lower right", fontsize=7,
              ncol=len(SEGMENTS), bbox_to_anchor=(1, 1.02), borderaxespad=0)

    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = f"{row['model']}_{row['dataset']}_dim{row['dim']}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

print(f"\nAll figures saved to {out_dir}/")
