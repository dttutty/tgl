#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import statistics
import sys


FINAL_TEST_AP_RE = re.compile(
    r"^\s*test ap:([0-9.]+)\s+test auc:[0-9.]+", re.IGNORECASE | re.MULTILINE
)


def seed_sort_key(path: pathlib.Path) -> tuple[int, str]:
    try:
        return (0, f"{int(path.stem.split('_', 1)[1]):08d}")
    except (IndexError, ValueError):
        return (1, path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build results.tsv / summary.tsv for TGL seed sweep directories."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more seed sweep run directories containing seed_*.log files.",
    )
    return parser.parse_args()


def summarize_run_dir(run_dir: pathlib.Path) -> dict[str, str]:
    logs = sorted(run_dir.glob("seed_*.log"), key=seed_sort_key)
    if not logs:
        raise SystemExit(f"No seed logs found in {run_dir}")

    results_rows: list[dict[str, str]] = []
    failed_logs: list[str] = []
    for log_path in logs:
        matches = FINAL_TEST_AP_RE.findall(
            log_path.read_text(encoding="utf-8", errors="ignore")
        )
        if not matches:
            failed_logs.append(log_path.name)
            continue
        results_rows.append(
            {
                "seed": log_path.stem.split("_", 1)[1],
                "test_ap": matches[-1],
                "log_path": str(log_path.resolve()),
            }
        )

    results_tsv = run_dir / "results.tsv"
    with results_tsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "test_ap", "log_path"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(results_rows)

    values = [float(row["test_ap"]) for row in results_rows]
    mean = statistics.fmean(values) if values else None
    stdev = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)

    if not results_rows:
        status = "no_valid_logs"
    elif failed_logs:
        status = "partial"
    else:
        status = "ok"

    summary_row = {
        "run_dir": str(run_dir.resolve()),
        "total_log_count": str(len(logs)),
        "parsed_log_count": str(len(results_rows)),
        "failed_log_count": str(len(failed_logs)),
        "mean_test_ap": f"{mean:.6f}" if mean is not None else "NA",
        "std_test_ap": f"{stdev:.6f}" if stdev is not None else "NA",
        "status": status,
        "failed_logs": ",".join(failed_logs),
    }

    summary_tsv = run_dir / "summary.tsv"
    with summary_tsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary_row.keys()), delimiter="\t"
        )
        writer.writeheader()
        writer.writerow(summary_row)

    summary_lines = [
        f"mean_test_ap={summary_row['mean_test_ap']}",
        f"std_test_ap={summary_row['std_test_ap']}",
        f"parsed_log_count={summary_row['parsed_log_count']}",
        f"failed_log_count={summary_row['failed_log_count']}",
        f"status={summary_row['status']}",
    ]
    if failed_logs:
        summary_lines.append(f"failed_logs={summary_row['failed_logs']}")
    (run_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return summary_row


def main() -> int:
    args = parse_args()
    exit_code = 0
    for run_dir_arg in args.run_dirs:
        run_dir = pathlib.Path(run_dir_arg).resolve()
        if not run_dir.is_dir():
            print(f"Skipping non-directory path: {run_dir}", file=sys.stderr)
            exit_code = 1
            continue
        try:
            summary_row = summarize_run_dir(run_dir)
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            exit_code = 1
            continue

        print(
            "\t".join(
                [
                    summary_row["run_dir"],
                    summary_row["mean_test_ap"],
                    summary_row["std_test_ap"],
                    summary_row["parsed_log_count"],
                    summary_row["failed_log_count"],
                    summary_row["status"],
                ]
            )
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
