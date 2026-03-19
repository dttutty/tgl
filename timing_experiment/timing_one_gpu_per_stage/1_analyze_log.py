#!/usr/bin/env python
"""
Parse benchmark timing logs and generate a CSV.

Usage:
    python exp/benchmark_timing/1_analyze_log.py [--log_dir exp/benchmark_timing/logs]

Output:
    exp/benchmark_timing/results.csv
"""
import argparse
import csv
import os
import re
import glob

EPOCH_PATTERN = re.compile(
    r"total time:([\d.]+)s\s+"
    r"sample:([\d.]+)s\s+"
    r"fetch feature:([\d.]+)s\s+"
    r"fetch memory:([\d.]+)s\s+"
    r"forward:([\d.]+)s\s+"
    r"backward:([\d.]+)s\s+"
    r"memory update:([\d.]+)s"
)

AVG_PATTERN = re.compile(
    r"Average over \d+ epochs:.*?"
    r"total time:([\d.]+)s\s+"
    r"sample:([\d.]+)s\s+"
    r"fetch feature:([\d.]+)s\s+"
    r"fetch memory:([\d.]+)s\s+"
    r"forward:([\d.]+)s\s+"
    r"backward:([\d.]+)s\s+"
    r"memory update:([\d.]+)s",
    re.DOTALL,
)

METRIC_PATTERN = re.compile(
    r"train loss:([\d.]+)\s+val ap:([\d.]+)\s+val auc:([\d.]+)"
)

TEST_PATTERN = re.compile(
    r"test AP:([\d.]+)\s+test (?:AUC|MRR):([\d.]+)"
)

# log filename: {MODEL}_{DATASET}_{pin_label}_bs{B}_memdim{N}_ep{E}.log
FILENAME_PATTERN = re.compile(
    r"^(?:[^_]+_)?([^_]+)_([^_]+)_(nopin|pin)_bs(\d+)_memdim(\d+)_ep(\d+)\.log$"
)

FIELDS = ["total", "sample", "fetch_feature", "fetch_memory", "forward", "backward", "memory_update"]


def parse_log(filepath):
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

    with open(filepath, "r") as f:
        content = f.read()

    rows = []

    epoch_matches = EPOCH_PATTERN.findall(content)
    metric_matches = METRIC_PATTERN.findall(content)
    for i, em in enumerate(epoch_matches):
        row = {
            "model": model,
            "dataset": dataset,
            "pin_memory": pin_label,
            "batch_size": batch_size,
            "dim": dim,
            "epochs": epochs,
            "epoch": i,
        }
        for j, field in enumerate(FIELDS):
            row[field] = float(em[j])
        if i < len(metric_matches):
            row["train_loss"] = float(metric_matches[i][0])
            row["val_ap"]     = float(metric_matches[i][1])
            row["val_auc"]    = float(metric_matches[i][2])
        rows.append(row)

    avg_match = AVG_PATTERN.search(content)
    if avg_match:
        row = {
            "model": model,
            "dataset": dataset,
            "pin_memory": pin_label,
            "batch_size": batch_size,
            "dim": dim,
            "epochs": epochs,
            "epoch": "avg",
        }
        for j, field in enumerate(FIELDS):
            row[field] = float(avg_match.group(j + 1))
        test_match = TEST_PATTERN.search(content)
        if test_match:
            row["test_ap"]      = float(test_match.group(1))
            row["test_auc_mrr"] = float(test_match.group(2))
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")

    log_files = sorted(glob.glob(os.path.join(args.log_dir, "*.log")))
    if not log_files:
        print(f"No .log files found in {args.log_dir}")
        return

    all_rows = []
    for lf in log_files:
        rows = parse_log(lf)
        if rows:
            print(f"  Parsed {lf}: {len(rows)} rows")
        all_rows.extend(rows)

    csv_fields = ["model", "dataset", "pin_memory", "batch_size", "dim", "epochs", "epoch"] + FIELDS + \
                 ["train_loss", "val_ap", "val_auc", "test_ap", "test_auc_mrr"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{len(all_rows)} rows saved to {args.output}")


if __name__ == "__main__":
    main()
