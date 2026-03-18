#!/usr/bin/env python3
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path  = os.path.join(script_dir, "results.csv")
output_path = os.path.join(script_dir, "results_lastfm_rank0.csv")

df = pd.read_csv(input_path)
filtered = df[(df["dataset"] == "LASTFM") & (df["rank"] == 0)]
filtered.to_csv(output_path, index=False)
print(f"Saved {len(filtered)} rows to {output_path}")
