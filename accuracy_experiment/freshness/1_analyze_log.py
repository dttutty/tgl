import argparse
import csv
import pathlib
import re


FILENAME_RE = re.compile(
    r"^(?P<model>[^_]+)_(?P<dataset>[^_]+)_bs(?P<batch_size>\d+)_memdim(?P<dim_out>\d+)_ep(?P<epochs>\d+)_delay(?P<delay>\d+)_(?:run(?P<run_id>\d+)_)?pin\.log$"
)
METRIC_RE = re.compile(r"test AP:([0-9.eE+-]+)\s+test (AUC|MRR):([0-9.eE+-]+)")


def detect_status(text: str, metric_match: re.Match[str] | None) -> str:
    if metric_match is not None:
        return "ok"
    if "Traceback (most recent call last):" in text:
        return "failed"
    return "missing_metrics"


def parse_log(log_path: pathlib.Path) -> dict | None:
    filename_match = FILENAME_RE.match(log_path.name)
    if filename_match is None:
        return None

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    metric_matches = METRIC_RE.findall(text)
    metric_match = metric_matches[-1] if metric_matches else None
    status = detect_status(text, metric_match)

    row = filename_match.groupdict()
    if row["run_id"] is None:
        row["run_id"] = "1"
    row["log_file"] = str(log_path)
    if metric_match is None:
        row["test_ap"] = "NA"
        row["test_metric"] = "NA"
        row["test_score"] = "NA"
    else:
        row["test_ap"] = metric_match[0]
        row["test_metric"] = metric_match[1]
        row["test_score"] = metric_match[2]
    row["status"] = status
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze freshness experiment logs.")
    parser.add_argument(
        "--log_dir",
        default=str(pathlib.Path(__file__).resolve().parent / "logs"),
        help="directory containing freshness log files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output TSV path; defaults to <log_dir>/summary.tsv",
    )
    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir).resolve()
    output_path = pathlib.Path(args.output).resolve() if args.output else log_dir / "summary.tsv"

    rows = []
    for log_path in sorted(log_dir.glob("*.log")):
        row = parse_log(log_path)
        if row is not None:
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "model",
            "dataset",
            "batch_size",
            "dim_out",
            "epochs",
            "delay",
            "run_id",
            "test_ap",
            "test_metric",
            "test_score",
            "status",
            "log_file",
        ])
        for row in rows:
            writer.writerow([
                row["model"],
                row["dataset"],
                row["batch_size"],
                row["dim_out"],
                row["epochs"],
                row["delay"],
                row["run_id"],
                row["test_ap"],
                row["test_metric"],
                row["test_score"],
                row["status"],
                row["log_file"],
            ])

    print(f"Parsed {len(rows)} log files")
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
