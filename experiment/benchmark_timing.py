#!/usr/bin/env python
"""
Benchmark TGN, APAN, JODIE across 6 datasets and collect per-stage timing into a CSV.

Usage:
    python experiment/benchmark_timing.py [--gpu 0]

Output:
    experiment/timing_results.csv
"""
import argparse
import csv
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "TGN":   os.path.join(REPO_ROOT, "config", "TGN.yml"),
    "APAN":  os.path.join(REPO_ROOT, "config", "APAN.yml"),
    "JODIE": os.path.join(REPO_ROOT, "config", "JODIE.yml"),
}

# dataset -> extra args passed to train.py
DATASETS = {
    "LASTFM":    ["--rand_edge_features", "128"],
    "MOOC":      ["--rand_edge_features", "128"],
    "REDDIT":    [],
    "WIKIPEDIA": [],
    "GDELT":     [],
    "MAG":       [],
}

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

FIELDS = ["total", "sample", "fetch_feature", "fetch_memory", "forward", "backward", "memory_update"]


def run_model(model_name, dataset, config_path, extra_args, gpu, pin_memory=False):
    pin_label = "pin" if pin_memory else "nopin"
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "train.py"),
        "--data", dataset,
        "--config", config_path,
        "--gpu", str(gpu),
        "--model_name", f"{model_name}_{dataset}_{pin_label}",
    ] + extra_args
    if pin_memory:
        cmd.append("--pin_memory")

    print(f"\n{'='*60}")
    print(f"[{model_name} / {dataset} / {pin_label}] {' '.join(cmd)}")
    print(f"{'='*60}\n")

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    output = proc.stdout + proc.stderr
    print(output)

    if proc.returncode != 0:
        print(f"WARNING: {model_name}/{dataset} exited with code {proc.returncode}")

    return output


def parse_output(output, model_name, dataset, pin_memory=False):
    rows = []
    pin_label = "pin" if pin_memory else "nopin"

    epoch_matches = EPOCH_PATTERN.findall(output)
    metric_matches = METRIC_PATTERN.findall(output)
    for i, m in enumerate(epoch_matches):
        row = {"model": model_name, "dataset": dataset, "pin_memory": pin_label, "epoch": i}
        for j, field in enumerate(FIELDS):
            row[field] = float(m[j])
        if i < len(metric_matches):
            row["train_loss"] = float(metric_matches[i][0])
            row["val_ap"]     = float(metric_matches[i][1])
            row["val_auc"]    = float(metric_matches[i][2])
        rows.append(row)

    avg_match = AVG_PATTERN.search(output)
    if avg_match:
        row = {"model": model_name, "dataset": dataset, "pin_memory": pin_label, "epoch": "avg"}
        for j, field in enumerate(FIELDS):
            row[field] = float(avg_match.group(j + 1))
        test_match = TEST_PATTERN.search(output)
        if test_match:
            row["test_ap"]      = float(test_match.group(1))
            row["test_auc_mrr"] = float(test_match.group(2))
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    all_rows = []
    for dataset, extra_args in DATASETS.items():
        for model_name, config_path in MODELS.items():
            for pin in [False, True]:
                output = run_model(model_name, dataset, config_path, extra_args, args.gpu, pin_memory=pin)
                all_rows.extend(parse_output(output, model_name, dataset, pin_memory=pin))

    csv_path = os.path.join(REPO_ROOT, "experiment", "timing_results.csv")
    csv_fields = ["model", "dataset", "pin_memory", "epoch"] + FIELDS + ["train_loss", "val_ap", "val_auc", "test_ap", "test_auc_mrr"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
