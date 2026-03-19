#!/usr/bin/env python3
"""Parse distributed timing benchmark logs into a CSV.

Input logs are expected to contain lines like:
  [rank 0] forward total       : n=... avg=1.234ms p50=... p90=... p99=...
  [rank 0]   memory_update     : ...
  [rank 0]   get_emb (excl mem): ...
  [rank 0]   edge_predictor    : ...
  [rank 0] backward total      : ...
  [rank 0] bwd->first allreduce: ...
"""

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Optional

FILENAME_PATTERN = re.compile(
    r"^(?:[^_]+_)?([^_]+)_([^_]+)_(nopin|pin)_bs(\d+)_memdim(\d+)_ep(\d+)\.log$"
)

LINE_PATTERNS = {
    "forward_ms": re.compile(r"\[rank\s+(\d+)\]\s+forward total\s*:\s*.*?avg=([\d.]+)ms"),
    "memory_updater_ms": re.compile(r"\[rank\s+(\d+)\]\s+memory_update\s*:\s*.*?avg=([\d.]+)ms"),
    "encoder_ms": re.compile(r"\[rank\s+(\d+)\]\s+get_emb \(excl mem\)\s*:\s*.*?avg=([\d.]+)ms"),
    "predictor_ms": re.compile(r"\[rank\s+(\d+)\]\s+edge_predictor\s*:\s*.*?avg=([\d.]+)ms"),
    "backward_ms": re.compile(r"\[rank\s+(\d+)\]\s+backward total\s*:\s*.*?avg=([\d.]+)ms"),
    "bwd_to_first_ar_ms": re.compile(r"\[rank\s+(\d+)\]\s+bwd->first allreduce\s*:\s*.*?avg=([\d.]+)ms"),
}


def parse_log(filepath: str) -> List[Dict[str, object]]:
    filename = os.path.basename(filepath)
    m = FILENAME_PATTERN.match(filename)
    if not m:
        print(f"Skipping unrecognized file: {filename}")
        return []

    model, dataset, pin_label, batch_size, dim, epochs = (
        m.group(1),
        m.group(2),
        m.group(3),
        int(m.group(4)),
        m.group(5),
        int(m.group(6)),
    )
    dim_value: Optional[int] = int(dim) if dim is not None else None

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    per_rank: Dict[int, Dict[str, float]] = {}

    for field, pattern in LINE_PATTERNS.items():
        matches = pattern.findall(content)
        if not matches:
            continue
        for rank_str, val_str in matches:
            rank = int(rank_str)
            val = float(val_str)
            if rank not in per_rank:
                per_rank[rank] = {}
            # Keep last occurrence per field/rank in case log contains retries.
            per_rank[rank][field] = val

    rows: List[Dict[str, object]] = []
    for rank, values in sorted(per_rank.items(), key=lambda kv: kv[0]):
        row: Dict[str, object] = {
            "model": model,
            "dataset": dataset,
            "pin_memory": pin_label,
            "batch_size": batch_size,
            "dim": dim_value,
            "epochs": epochs,
            "rank": rank,
            "log_file": filename,
        }
        row.update(values)

        required = [
            "forward_ms",
            "memory_updater_ms",
            "encoder_ms",
            "predictor_ms",
            "backward_ms",
            "bwd_to_first_ar_ms",
        ]
        if all(k in row for k in required):
            rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--log_dir", type=str, default=os.path.join(script_dir, "logs"))
    parser.add_argument("--output", type=str, default=os.path.join(script_dir, "results.csv"))
    args = parser.parse_args()

    log_files = sorted(glob.glob(os.path.join(args.log_dir, "*.log")))
    if not log_files:
        print(f"No .log files found in {args.log_dir}")
        return

    all_rows: List[Dict[str, object]] = []
    for lf in log_files:
        rows = parse_log(lf)
        if rows:
            print(f"  Parsed {lf}: {len(rows)} rank rows")
            all_rows.extend(rows)

    if not all_rows:
        print("No valid timing rows parsed from logs.")
        return

    csv_fields = [
        "model",
        "dataset",
        "pin_memory",
        "batch_size",
        "dim",
        "epochs",
        "rank",
        "forward_ms",
        "memory_updater_ms",
        "encoder_ms",
        "predictor_ms",
        "backward_ms",
        "bwd_to_first_ar_ms",
        "log_file",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{len(all_rows)} rows saved to {args.output}")


if __name__ == "__main__":
    main()
