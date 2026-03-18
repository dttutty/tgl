#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

FILENAME_PATTERN = re.compile(
    r"^(?P<model>.+?)_(?P<dataset>.+?)_ngpu(?P<num_gpus>\d+)_memdim(?P<mem_dim>\d+)_ep(?P<epochs>\d+)_rep(?P<repeat>\d+)\.log$"
)
TEST_AP_PATTERN = re.compile(r"test ap:([0-9]*\.?[0-9]+)", flags=re.IGNORECASE)
TEST_AUC_PATTERN = re.compile(r"test auc:([0-9]*\.?[0-9]+)", flags=re.IGNORECASE)
BEST_EPOCH_PATTERN = re.compile(r"Best model at epoch (\d+)\.", flags=re.IGNORECASE)


def parse_one_log(log_path: Path, log_dir: Path) -> dict[str, object] | None:
    match = FILENAME_PATTERN.match(log_path.name)
    if match is None:
        print(f"Skipping unrecognized file: {log_path}")
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    ap_matches = TEST_AP_PATTERN.findall(text)
    if not ap_matches:
        print(f"Missing test AP in: {log_path}")
        return None

    auc_matches = TEST_AUC_PATTERN.findall(text)
    best_epoch_matches = BEST_EPOCH_PATTERN.findall(text)
    meta = match.groupdict()
    parent_rel = log_path.parent.relative_to(log_dir)
    run_tag = "." if str(parent_rel) == "." else str(parent_rel)

    return {
        "run_tag": run_tag,
        "model": meta["model"],
        "dataset": meta["dataset"],
        "num_gpus": int(meta["num_gpus"]),
        "mem_dim": int(meta["mem_dim"]),
        "epochs": int(meta["epochs"]),
        "repeat": int(meta["repeat"]),
        "best_epoch": int(best_epoch_matches[-1]) if best_epoch_matches else "",
        "test_ap": float(ap_matches[-1]),
        "test_auc": float(auc_matches[-1]) if auc_matches else "",
        "log_file": str(log_path.relative_to(log_dir)),
    }


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["run_tag"],
            row["model"],
            row["dataset"],
            row["num_gpus"],
            row["mem_dim"],
            row["epochs"],
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: int(row["repeat"]))
        ap_values = [float(row["test_ap"]) for row in group]
        auc_values = [float(row["test_auc"]) for row in group if row["test_auc"] != ""]
        summary_rows.append(
            {
                "run_tag": key[0],
                "model": key[1],
                "dataset": key[2],
                "num_gpus": key[3],
                "mem_dim": key[4],
                "epochs": key[5],
                "runs": len(group),
                "mean_test_ap": f"{statistics.mean(ap_values):.6f}",
                "std_test_ap": f"{statistics.stdev(ap_values):.6f}" if len(ap_values) > 1 else "0.000000",
                "min_test_ap": f"{min(ap_values):.6f}",
                "max_test_ap": f"{max(ap_values):.6f}",
                "mean_test_auc": f"{statistics.mean(auc_values):.6f}" if auc_values else "",
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_log_dir = script_dir / "logs"
    if not default_log_dir.exists():
        default_log_dir = script_dir / "log"

    parser = argparse.ArgumentParser(
        description="Parse compare_ap_tgl_vs_frost logs and summarize final test AP by mem_dim."
    )
    parser.add_argument("--log_dir", type=Path, default=default_log_dir)
    parser.add_argument("--output", type=Path, default=None, help="Per-run parsed CSV path.")
    parser.add_argument("--summary_output", type=Path, default=None, help="Aggregated summary CSV path.")
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    output = args.output.resolve() if args.output else log_dir / "results.csv"
    summary_output = args.summary_output.resolve() if args.summary_output else log_dir / "summary.csv"

    if not log_dir.exists():
        raise SystemExit(f"log_dir does not exist: {log_dir}")

    log_files = sorted(path for path in log_dir.rglob("*.log") if path.is_file())
    if not log_files:
        print(f"No .log files found under {log_dir}")
        return

    rows: list[dict[str, object]] = []
    for log_file in log_files:
        row = parse_one_log(log_file, log_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No valid AP results parsed from logs.")
        return

    rows.sort(key=lambda row: (str(row["run_tag"]), int(row["mem_dim"]), int(row["repeat"])))
    summary_rows = build_summary(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    write_csv(
        output,
        rows,
        [
            "run_tag",
            "model",
            "dataset",
            "num_gpus",
            "mem_dim",
            "epochs",
            "repeat",
            "best_epoch",
            "test_ap",
            "test_auc",
            "log_file",
        ],
    )
    write_csv(
        summary_output,
        summary_rows,
        [
            "run_tag",
            "model",
            "dataset",
            "num_gpus",
            "mem_dim",
            "epochs",
            "runs",
            "mean_test_ap",
            "std_test_ap",
            "min_test_ap",
            "max_test_ap",
            "mean_test_auc",
        ],
    )

    print(f"Parsed {len(rows)} log(s)")
    print(f"Per-run results: {output}")
    print(f"Summary: {summary_output}")
    print("")
    print("run_tag,mem_dim,runs,mean_test_ap,std_test_ap,min_test_ap,max_test_ap")
    for row in summary_rows:
        print(
            f"{row['run_tag']},{row['mem_dim']},{row['runs']},{row['mean_test_ap']},{row['std_test_ap']},{row['min_test_ap']},{row['max_test_ap']}"
        )


if __name__ == "__main__":
    main()
