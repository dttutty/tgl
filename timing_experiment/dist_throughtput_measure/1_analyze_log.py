#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import re
from typing import Dict, List

FILENAME_PATTERN = re.compile(
    r"^(?:(?P<username>[^_]+)(?:_(?P<hostname>[^_]+))?_)?(?P<model>[^_]+)_(?P<dataset>[^_]+)_bs(?P<batch_size>\d+)_ngpu(?P<num_gpus>\d+)_memdim(?P<mem_dim>\d+)_ep(?P<epochs>\d+)_rep(?P<repeat>\d+)\.log$"
)

SUMMARY_PATTERNS = {
    "gpu_stream": re.compile(
        r"\[rank\s+(\d+)\]\s+minibatch start->start \(GPU stream\):\s*"
        r"n=(\d+)\s+avg=([\d.]+)ms\s+p50=([\d.]+)ms\s+p90=([\d.]+)ms\s+p99=([\d.]+)ms"
    ),
    "wall": re.compile(
        r"\[rank\s+(\d+)\]\s+minibatch start->start \(wall\):\s*"
        r"n=(\d+)\s+avg=([\d.]+)ms\s+p50=([\d.]+)ms\s+p90=([\d.]+)ms\s+p99=([\d.]+)ms"
    ),
}


def ms_to_eps(batch_size: int, avg_ms: float) -> float:
    if avg_ms <= 0:
        return 0.0
    return batch_size * 1000.0 / avg_ms


def parse_log(filepath: str) -> List[Dict[str, object]]:
    filename = os.path.basename(filepath)
    filename_match = FILENAME_PATTERN.match(filename)
    if not filename_match:
        print(f"Skipping unrecognized file: {filename}")
        return []

    meta = filename_match.groupdict()
    batch_size = int(meta["batch_size"])
    num_gpus = int(meta["num_gpus"])

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    per_rank: Dict[int, Dict[str, object]] = {}
    for metric_name, pattern in SUMMARY_PATTERNS.items():
        for match in pattern.findall(content):
            rank = int(match[0])
            interval_count = int(match[1])
            avg_ms = float(match[2])
            p50_ms = float(match[3])
            p90_ms = float(match[4])
            p99_ms = float(match[5])

            row = per_rank.setdefault(rank, {})
            row[f"{metric_name}_interval_count"] = interval_count
            row[f"{metric_name}_avg_ms"] = avg_ms
            row[f"{metric_name}_p50_ms"] = p50_ms
            row[f"{metric_name}_p90_ms"] = p90_ms
            row[f"{metric_name}_p99_ms"] = p99_ms
            row[f"per_gpu_{metric_name}_eps"] = ms_to_eps(batch_size, avg_ms)
            row[f"global_{metric_name}_eps"] = row[f"per_gpu_{metric_name}_eps"] * num_gpus

    rows: List[Dict[str, object]] = []
    for rank, values in sorted(per_rank.items(), key=lambda item: item[0]):
        row: Dict[str, object] = {
            "model": meta["model"],
            "dataset": meta["dataset"],
            "batch_size": batch_size,
            "num_gpus": num_gpus,
            "mem_dim": int(meta["mem_dim"]),
            "epochs": int(meta["epochs"]),
            "repeat": int(meta["repeat"]),
            "rank": rank,
            "log_file": filename,
        }
        row.update(values)

        if "per_gpu_wall_eps" in row or "per_gpu_gpu_stream_eps" in row:
            rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse distributed throughput logs and export one CSV row per worker rank."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--log_dir", type=str, default=os.path.join(script_dir, "logs"))
    parser.add_argument("--output", type=str, default=os.path.join(script_dir, "results.csv"))
    args = parser.parse_args()

    log_files = sorted(glob.glob(os.path.join(args.log_dir, "*.log")))
    if not log_files:
        print(f"No .log files found in {args.log_dir}")
        return

    all_rows: List[Dict[str, object]] = []
    for log_file in log_files:
        rows = parse_log(log_file)
        if rows:
            print(f"  Parsed {log_file}: {len(rows)} rank rows")
            all_rows.extend(rows)

    if not all_rows:
        print("No valid throughput rows parsed from logs.")
        return

    fieldnames = [
        "model",
        "dataset",
        "batch_size",
        "num_gpus",
        "mem_dim",
        "epochs",
        "repeat",
        "rank",
        "gpu_stream_interval_count",
        "gpu_stream_avg_ms",
        "gpu_stream_p50_ms",
        "gpu_stream_p90_ms",
        "gpu_stream_p99_ms",
        "per_gpu_gpu_stream_eps",
        "global_gpu_stream_eps",
        "wall_interval_count",
        "wall_avg_ms",
        "wall_p50_ms",
        "wall_p90_ms",
        "wall_p99_ms",
        "per_gpu_wall_eps",
        "global_wall_eps",
        "log_file",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{len(all_rows)} rows saved to {args.output}")


if __name__ == "__main__":
    main()
