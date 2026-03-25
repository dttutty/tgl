import argparse
import csv
import math
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from scipy.stats import t


FILENAME_RE = re.compile(
    r"^(?:(?P<username>[^_]+)(?:_(?P<hostname>[^_]+))?_)?(?P<model>[^_]+)_(?P<dataset>[^_]+)_bs(?P<batch_size>\d+)_memdim(?P<dim_out>\d+)_ep(?P<epochs>\d+)_delay(?P<delay>\d+)_(?:run(?P<run_id>\d+)_)?pin\.log$"
)
METRIC_RE = re.compile(r"test AP:([0-9.eE+-]+)\s+test (AUC|MRR):([0-9.eE+-]+)")

SUMMARY_COLUMNS = [
    "username",
    "hostname",
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
]
STATS_GROUP_COLUMNS = [
    "model",
    "dataset",
    "username",
    "hostname",
    "batch_size",
    "dim_out",
    "epochs",
    "delay",
    "test_metric",
]


def detect_status(text: str, has_metrics: bool) -> str:
    if has_metrics:
        return "ok"
    if "Traceback (most recent call last):" in text:
        return "failed"
    return "missing_metrics"


def parse_log(log_path: pathlib.Path) -> Optional[Dict[str, str]]:
    filename_match = FILENAME_RE.match(log_path.name)
    if filename_match is None:
        return None

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    metric_matches = METRIC_RE.findall(text)
    metric_match = metric_matches[-1] if metric_matches else None

    row = filename_match.groupdict()
    row["username"] = row["username"] or "unknown"
    row["hostname"] = row["hostname"] or "unknown"
    row["run_id"] = row["run_id"] or "1"
    row["log_file"] = str(log_path)
    row["status"] = detect_status(text, metric_match is not None)
    if metric_match is None:
        row["test_ap"] = "NA"
        row["test_metric"] = "NA"
        row["test_score"] = "NA"
    else:
        row["test_ap"] = metric_match[0]
        row["test_metric"] = metric_match[1]
        row["test_score"] = metric_match[2]
    return row


def collect_log_rows(log_dir: pathlib.Path) -> List[Dict[str, str]]:
    rows = []
    for log_path in sorted(log_dir.glob("*.log")):
        row = parse_log(log_path)
        if row is not None:
            rows.append(row)
    return rows


def write_summary(rows: List[Dict[str, str]], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(SUMMARY_COLUMNS)
        for row in rows:
            writer.writerow([row.get(column, "") for column in SUMMARY_COLUMNS])


def infer_identity_from_log_file(log_file: str) -> Tuple[str, str]:
    filename_match = FILENAME_RE.match(pathlib.Path(log_file).name)
    if filename_match is None:
        return ("unknown", "unknown")
    return (
        filename_match.group("username") or "unknown",
        filename_match.group("hostname") or "unknown",
    )


def fill_identity_column(
    df: pd.DataFrame, column: str, inferred_values: pd.Series
) -> pd.Series:
    if column not in df.columns:
        return inferred_values

    values = df[column].fillna("").astype(str).str.strip()
    missing_mask = values == ""
    values.loc[missing_mask] = inferred_values.loc[missing_mask]
    return values


def normalize_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    if "log_file" not in normalized.columns:
        normalized["log_file"] = ""

    inferred_username = normalized["log_file"].apply(
        lambda value: infer_identity_from_log_file(str(value))[0]
    )
    inferred_hostname = normalized["log_file"].apply(
        lambda value: infer_identity_from_log_file(str(value))[1]
    )

    normalized["username"] = fill_identity_column(
        normalized, "username", inferred_username
    )
    normalized["hostname"] = fill_identity_column(
        normalized, "hostname", inferred_hostname
    )

    for column in ["model", "dataset", "test_metric", "status"]:
        if column not in normalized.columns:
            normalized[column] = "unknown"
        else:
            normalized[column] = (
                normalized[column].fillna("unknown").astype(str).str.strip()
            )

    for column in ["batch_size", "dim_out", "epochs", "delay", "run_id"]:
        if column not in normalized.columns:
            normalized[column] = -1
        normalized[column] = (
            pd.to_numeric(normalized[column], errors="coerce")
            .fillna(-1)
            .astype(int)
        )

    if "test_ap" not in normalized.columns:
        normalized["test_ap"] = float("nan")
    if "test_score" not in normalized.columns:
        normalized["test_score"] = float("nan")
    normalized["test_ap"] = pd.to_numeric(normalized["test_ap"], errors="coerce")
    normalized["test_score"] = pd.to_numeric(
        normalized["test_score"], errors="coerce"
    )
    return normalized


def ci95_halfwidth(var: float, n_runs: int) -> float:
    if pd.isna(var) or n_runs <= 1:
        return float("nan")
    se = math.sqrt(var / n_runs)
    return t.ppf(0.975, df=n_runs - 1) * se


def build_stats_table(
    df: pd.DataFrame, expected_runs: Optional[int] = None
) -> pd.DataFrame:
    normalized = normalize_summary_df(df)
    valid = normalized[
        (normalized["status"] == "ok")
        & normalized["test_ap"].notna()
        & normalized["test_score"].notna()
    ].copy()

    if valid.empty:
        stats = pd.DataFrame(
            columns=[
                "model",
                "dataset",
                "username",
                "hostname",
                "batch_size",
                "dim_out",
                "epochs",
                "delay",
                "test_metric",
                "n_runs",
                "ap_mean",
                "ap_std",
                "ap_var",
                "score_mean",
                "score_std",
                "score_var",
                "ap_ci95_halfwidth",
                "score_ci95_halfwidth",
                "auc_mean",
                "auc_std",
                "auc_var",
                "auc_ci95_halfwidth",
                "expected_runs",
                "matches_expected_runs",
            ]
        )
    else:
        stats = (
            valid.groupby(STATS_GROUP_COLUMNS, as_index=False)
            .agg(
                n_runs=("run_id", "count"),
                ap_mean=("test_ap", "mean"),
                ap_std=("test_ap", "std"),
                ap_var=("test_ap", "var"),
                score_mean=("test_score", "mean"),
                score_std=("test_score", "std"),
                score_var=("test_score", "var"),
            )
            .sort_values(STATS_GROUP_COLUMNS)
        )

        stats["ap_ci95_halfwidth"] = stats.apply(
            lambda row: ci95_halfwidth(row["ap_var"], int(row["n_runs"])),
            axis=1,
        )
        stats["score_ci95_halfwidth"] = stats.apply(
            lambda row: ci95_halfwidth(row["score_var"], int(row["n_runs"])),
            axis=1,
        )

        for column in [
            "ap_mean",
            "ap_std",
            "ap_var",
            "score_mean",
            "score_std",
            "score_var",
            "ap_ci95_halfwidth",
            "score_ci95_halfwidth",
        ]:
            stats[column] = stats[column].round(6)

        stats["auc_mean"] = stats["score_mean"]
        stats["auc_std"] = stats["score_std"]
        stats["auc_var"] = stats["score_var"]
        stats["auc_ci95_halfwidth"] = stats["score_ci95_halfwidth"]

    if expected_runs is None:
        expected_runs = int(stats["n_runs"].max()) if not stats.empty else 0
    stats["expected_runs"] = expected_runs
    stats["matches_expected_runs"] = stats["n_runs"] == stats["expected_runs"]
    return stats


def main() -> None:
    default_log_dir = pathlib.Path(__file__).resolve().parent / "logs"
    default_summary_path = default_log_dir / "summary.tsv"
    default_stats_path = default_log_dir / "summary_stats_by_dim_delay.tsv"

    parser = argparse.ArgumentParser(
        description="Aggregate freshness experiment results. Default mode scans raw log files, rebuilds summary.tsv, then writes aggregated stats."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--from-summary",
        dest="from_summary",
        action="store_true",
        help="read an existing summary.tsv and aggregate it",
    )
    mode_group.add_argument(
        "--from-logs",
        dest="from_summary",
        action="store_false",
        help="scan raw .log files, rebuild summary.tsv, then aggregate it (default)",
    )
    parser.set_defaults(from_summary=False)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=default_summary_path,
        help="summary.tsv input path in --from-summary mode",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=default_stats_path,
        help="aggregated stats TSV output path",
    )
    parser.add_argument(
        "--log_dir",
        type=pathlib.Path,
        default=default_log_dir,
        help="directory containing freshness log files in --from-logs mode",
    )
    parser.add_argument(
        "--summary-output",
        type=pathlib.Path,
        default=default_summary_path,
        help="summary.tsv output path in --from-logs mode",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=None,
        help="expected number of runs per group; defaults to the largest observed n_runs",
    )
    args = parser.parse_args()

    if args.from_summary:
        input_path = args.input.resolve()
        if not input_path.is_file():
            raise SystemExit(
                "summary.tsv not found: {}. Run with --from-logs first if you need to rebuild it.".format(
                    input_path
                )
            )
        df = pd.read_csv(str(input_path), sep="\t")
        stats = build_stats_table(df, expected_runs=args.expected_runs)
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(str(output_path), sep="\t", index=False)

        if not stats.empty:
            print(stats.to_string(index=False))
            print("")
        print("Loaded summary from {}".format(input_path))
        print("Aggregated {} groups".format(len(stats)))
        print("Stats written to {}".format(output_path))
        return

    log_dir = args.log_dir.resolve()
    summary_output_path = args.summary_output.resolve()
    output_path = args.output.resolve()

    rows = collect_log_rows(log_dir)
    write_summary(rows, summary_output_path)
    stats = build_stats_table(
        pd.DataFrame(rows, columns=SUMMARY_COLUMNS), expected_runs=args.expected_runs
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(str(output_path), sep="\t", index=False)

    if not stats.empty:
        print(stats.to_string(index=False))
        print("")
    print("Parsed {} log files".format(len(rows)))
    print("Summary written to {}".format(summary_output_path))
    print("Aggregated {} groups".format(len(stats)))
    print("Stats written to {}".format(output_path))


if __name__ == "__main__":
    main()
