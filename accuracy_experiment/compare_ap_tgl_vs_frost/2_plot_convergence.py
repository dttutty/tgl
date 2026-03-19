#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FILENAME_PATTERN = re.compile(
    r"^(?:(?P<username>[^_]+)(?:_(?P<hostname>[^_]+))?_)?(?P<model>[^_]+)_(?P<dataset>[^_]+)_bs(?P<batch_size>\d+)_ngpu(?P<num_gpus>\d+)_memdim(?P<mem_dim>\d+)_ep(?P<epochs>\d+)_rep(?P<repeat>\d+)\.log$"
)
EPOCH_PATTERN = re.compile(r"^Epoch (\d+):")
VAL_PATTERN = re.compile(r"\[VAL\]\s+Global AP:\s*([0-9]*\.?[0-9]+)\s+AUC:\s*([0-9]*\.?[0-9]+)")
TEST_PATTERN = re.compile(r"\[TEST\]\s+Global AP:\s*([0-9]*\.?[0-9]+)\s+AUC:\s*([0-9]*\.?[0-9]+)")
TRAIN_PATTERN = re.compile(
    r"train loss:([0-9]*\.?[0-9]+)\s+val ap:([0-9]*\.?[0-9]+)\s+val auc:([0-9]*\.?[0-9]+)",
    flags=re.IGNORECASE,
)
BEST_EPOCH_PATTERN = re.compile(r"Best model at epoch (\d+)\.", flags=re.IGNORECASE)
FINAL_TEST_PATTERN = re.compile(r"test ap:([0-9]*\.?[0-9]+)\s+test auc:([0-9]*\.?[0-9]+)", flags=re.IGNORECASE)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float | None = None
    val_ap: float | None = None
    val_auc: float | None = None
    test_ap: float | None = None
    test_auc: float | None = None


@dataclass
class ParsedLog:
    log_path: Path
    model: str
    dataset: str
    batch_size: int
    num_gpus: int
    mem_dim: int
    configured_epochs: int
    repeat: int
    epochs: list[EpochMetrics]
    best_epoch: int | None
    best_test_ap: float | None
    best_test_auc: float | None


def parse_one_log(log_path: Path) -> ParsedLog:
    match = FILENAME_PATTERN.match(log_path.name)
    if match is None:
        raise ValueError(f"Unrecognized log filename: {log_path.name}")

    meta = match.groupdict()
    text = log_path.read_text(encoding="utf-8", errors="replace")

    by_epoch: dict[int, EpochMetrics] = {}
    current_epoch: int | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        epoch_match = EPOCH_PATTERN.match(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            by_epoch.setdefault(current_epoch, EpochMetrics(epoch=current_epoch))
            continue

        if current_epoch is None:
            continue

        val_match = VAL_PATTERN.search(line)
        if val_match:
            epoch_metrics = by_epoch[current_epoch]
            epoch_metrics.val_ap = float(val_match.group(1))
            epoch_metrics.val_auc = float(val_match.group(2))
            continue

        test_match = TEST_PATTERN.search(line)
        if test_match:
            epoch_metrics = by_epoch[current_epoch]
            epoch_metrics.test_ap = float(test_match.group(1))
            epoch_metrics.test_auc = float(test_match.group(2))
            continue

        train_match = TRAIN_PATTERN.search(line)
        if train_match:
            epoch_metrics = by_epoch[current_epoch]
            epoch_metrics.train_loss = float(train_match.group(1))
            if epoch_metrics.val_ap is None:
                epoch_metrics.val_ap = float(train_match.group(2))
            if epoch_metrics.val_auc is None:
                epoch_metrics.val_auc = float(train_match.group(3))

    best_epoch_matches = BEST_EPOCH_PATTERN.findall(text)
    final_test_matches = FINAL_TEST_PATTERN.findall(text)

    return ParsedLog(
        log_path=log_path,
        model=meta["model"],
        dataset=meta["dataset"],
        batch_size=int(meta["batch_size"]),
        num_gpus=int(meta["num_gpus"]),
        mem_dim=int(meta["mem_dim"]),
        configured_epochs=int(meta["epochs"]),
        repeat=int(meta["repeat"]),
        epochs=[by_epoch[key] for key in sorted(by_epoch)],
        best_epoch=int(best_epoch_matches[-1]) if best_epoch_matches else None,
        best_test_ap=float(final_test_matches[-1][0]) if final_test_matches else None,
        best_test_auc=float(final_test_matches[-1][1]) if final_test_matches else None,
    )


def metric_series(epochs: list[EpochMetrics], field: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for epoch_metrics in epochs:
        value = getattr(epoch_metrics, field)
        if value is None:
            continue
        xs.append(epoch_metrics.epoch)
        ys.append(value)
    return xs, ys


def peak_epoch(epochs: list[EpochMetrics], field: str) -> tuple[int | None, float | None]:
    best_epoch: int | None = None
    best_value: float | None = None
    for epoch_metrics in epochs:
        value = getattr(epoch_metrics, field)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_epoch = epoch_metrics.epoch
            best_value = value
    return best_epoch, best_value


def value_at_epoch(epochs: list[EpochMetrics], epoch: int, field: str) -> float | None:
    for epoch_metrics in epochs:
        if epoch_metrics.epoch == epoch:
            return getattr(epoch_metrics, field)
    return None


def plot_one_log(parsed: ParsedLog, output_path: Path, dpi: int) -> dict[str, object]:
    if not parsed.epochs:
        raise ValueError(f"No epoch metrics parsed from {parsed.log_path}")

    train_x, train_y = metric_series(parsed.epochs, "train_loss")
    val_ap_x, val_ap_y = metric_series(parsed.epochs, "val_ap")
    test_ap_x, test_ap_y = metric_series(parsed.epochs, "test_ap")
    val_auc_x, val_auc_y = metric_series(parsed.epochs, "val_auc")
    test_auc_x, test_auc_y = metric_series(parsed.epochs, "test_auc")

    peak_val_ap_epoch, peak_val_ap = peak_epoch(parsed.epochs, "val_ap")
    last_epoch = parsed.epochs[-1].epoch
    chosen_best_epoch = parsed.best_epoch if parsed.best_epoch is not None else peak_val_ap_epoch
    chosen_best_label = "best_epoch" if parsed.best_epoch is not None else "peak_val_ap_epoch"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    ax_loss, ax_ap, ax_auc = axes

    if train_x:
        ax_loss.plot(train_x, train_y, color="#1f77b4", linewidth=1.8)
    ax_loss.set_ylabel("Train Loss")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if val_ap_x:
        ax_ap.plot(val_ap_x, val_ap_y, label="Val AP", color="#2ca02c", linewidth=1.8)
    if test_ap_x:
        ax_ap.plot(test_ap_x, test_ap_y, label="Test AP", color="#ff7f0e", linewidth=1.6, alpha=0.9)
    ax_ap.set_ylabel("AP")
    ax_ap.grid(True, alpha=0.25)
    ax_ap.legend(loc="lower right")

    if val_auc_x:
        ax_auc.plot(val_auc_x, val_auc_y, label="Val AUC", color="#17becf", linewidth=1.8)
    if test_auc_x:
        ax_auc.plot(test_auc_x, test_auc_y, label="Test AUC", color="#d62728", linewidth=1.6, alpha=0.9)
    ax_auc.set_ylabel("AUC")
    ax_auc.set_xlabel("Epoch")
    ax_auc.grid(True, alpha=0.25)
    ax_auc.legend(loc="lower right")

    if chosen_best_epoch is not None:
        for ax in axes:
            ax.axvline(chosen_best_epoch, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)

        best_val_ap = value_at_epoch(parsed.epochs, chosen_best_epoch, "val_ap")
        if best_val_ap is not None:
            ax_ap.scatter([chosen_best_epoch], [best_val_ap], color="#444444", s=28, zorder=3)

    title_left = (
        f"{parsed.model} / {parsed.dataset} / mem_dim={parsed.mem_dim} / repeat={parsed.repeat} / "
        f"batch={parsed.batch_size} / ngpu={parsed.num_gpus}"
    )
    title_right = (
        f"logged_epochs={len(parsed.epochs)} last_epoch={last_epoch} "
        f"{chosen_best_label}={chosen_best_epoch if chosen_best_epoch is not None else 'NA'}"
    )
    subtitle = (
        f"peak_val_ap={peak_val_ap:.4f}@{peak_val_ap_epoch} "
        if peak_val_ap is not None and peak_val_ap_epoch is not None
        else ""
    )
    if parsed.best_test_ap is not None and parsed.best_test_auc is not None:
        subtitle += f"best_test_ap={parsed.best_test_ap:.4f} best_test_auc={parsed.best_test_auc:.4f}"
    elif parsed.epochs[-1].test_ap is not None and parsed.epochs[-1].test_auc is not None:
        subtitle += f"last_test_ap={parsed.epochs[-1].test_ap:.4f} last_test_auc={parsed.epochs[-1].test_auc:.4f}"

    fig.suptitle(f"{title_left}\n{title_right}\n{subtitle}".strip(), fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return {
        "log_file": parsed.log_path.name,
        "output_file": str(output_path.name),
        "configured_epochs": parsed.configured_epochs,
        "logged_epochs": len(parsed.epochs),
        "last_epoch": last_epoch,
        "best_epoch": parsed.best_epoch if parsed.best_epoch is not None else "",
        "peak_val_ap_epoch": peak_val_ap_epoch if peak_val_ap_epoch is not None else "",
        "peak_val_ap": f"{peak_val_ap:.6f}" if peak_val_ap is not None else "",
        "last_train_loss": f"{parsed.epochs[-1].train_loss:.6f}" if parsed.epochs[-1].train_loss is not None else "",
        "last_val_ap": f"{parsed.epochs[-1].val_ap:.6f}" if parsed.epochs[-1].val_ap is not None else "",
        "last_test_ap": f"{parsed.epochs[-1].test_ap:.6f}" if parsed.epochs[-1].test_ap is not None else "",
        "last_val_auc": f"{parsed.epochs[-1].val_auc:.6f}" if parsed.epochs[-1].val_auc is not None else "",
        "last_test_auc": f"{parsed.epochs[-1].test_auc:.6f}" if parsed.epochs[-1].test_auc is not None else "",
        "best_test_ap": f"{parsed.best_test_ap:.6f}" if parsed.best_test_ap is not None else "",
        "best_test_auc": f"{parsed.best_test_auc:.6f}" if parsed.best_test_auc is not None else "",
        "status": "complete" if parsed.best_epoch is not None else "incomplete",
    }


def write_summary_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "log_file",
        "output_file",
        "configured_epochs",
        "logged_epochs",
        "last_epoch",
        "best_epoch",
        "peak_val_ap_epoch",
        "peak_val_ap",
        "last_train_loss",
        "last_val_ap",
        "last_test_ap",
        "last_val_auc",
        "last_test_auc",
        "best_test_ap",
        "best_test_auc",
        "status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_log_dir = script_dir / "logs"
    default_output_dir = default_log_dir / "plots"

    parser = argparse.ArgumentParser(description="Plot per-epoch convergence curves from compare_ap_tgl_vs_frost logs.")
    parser.add_argument("--log_dir", type=Path, default=default_log_dir, help="Directory containing .log files.")
    parser.add_argument("--output_dir", type=Path, default=default_output_dir, help="Directory for generated plot images.")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI for PNG output.")
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not log_dir.exists():
        raise SystemExit(f"log_dir does not exist: {log_dir}")

    log_files = sorted(path for path in log_dir.glob("*.log") if path.is_file())
    if not log_files:
        raise SystemExit(f"No .log files found in {log_dir}")

    summary_rows: list[dict[str, object]] = []
    for log_file in log_files:
        parsed = parse_one_log(log_file)
        output_path = output_dir / f"{log_file.stem}.png"
        row = plot_one_log(parsed, output_path, dpi=args.dpi)
        summary_rows.append(row)
        print(f"Wrote {output_path}")

    summary_path = output_dir / "convergence_summary.csv"
    write_summary_csv(summary_path, summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
