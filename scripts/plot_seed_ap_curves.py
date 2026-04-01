#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    val_ap: float | None = None
    test_ap: float | None = None


@dataclass
class ParsedLog:
    log_path: Path
    epochs: list[EpochMetrics]
    best_epoch: int | None
    best_test_ap: float | None


def parse_log(log_path: Path) -> ParsedLog:
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
            by_epoch[current_epoch].val_ap = float(val_match.group(1))
            continue

        test_match = TEST_PATTERN.search(line)
        if test_match:
            by_epoch[current_epoch].test_ap = float(test_match.group(1))
            continue

        train_match = TRAIN_PATTERN.search(line)
        if train_match and by_epoch[current_epoch].val_ap is None:
            by_epoch[current_epoch].val_ap = float(train_match.group(2))

    best_epoch_matches = BEST_EPOCH_PATTERN.findall(text)
    final_test_matches = FINAL_TEST_PATTERN.findall(text)

    return ParsedLog(
        log_path=log_path,
        epochs=[by_epoch[key] for key in sorted(by_epoch)],
        best_epoch=int(best_epoch_matches[-1]) if best_epoch_matches else None,
        best_test_ap=float(final_test_matches[-1][0]) if final_test_matches else None,
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


def plot_log(parsed: ParsedLog, output_path: Path, dpi: int) -> None:
    if not parsed.epochs:
        raise ValueError(f"No epoch metrics found in {parsed.log_path}")

    val_x, val_y = metric_series(parsed.epochs, "val_ap")
    test_x, test_y = metric_series(parsed.epochs, "test_ap")
    peak_val_epoch, peak_val_ap = peak_epoch(parsed.epochs, "val_ap")

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    if val_x:
        ax.plot(val_x, val_y, label="Val AP", color="#2ca02c", linewidth=2.0, marker="o", markersize=3)
    if test_x:
        ax.plot(test_x, test_y, label="Test AP", color="#ff7f0e", linewidth=1.8, marker="s", markersize=3, alpha=0.9)

    if parsed.best_epoch is not None:
        ax.axvline(parsed.best_epoch, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8, label=f"Best epoch {parsed.best_epoch}")

    if peak_val_epoch is not None and peak_val_ap is not None:
        ax.scatter([peak_val_epoch], [peak_val_ap], color="#2ca02c", s=30, zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.set_title(
        "\n".join(
            part
            for part in [
                f"{parsed.log_path.stem}: Val/Test AP vs Epoch",
                (
                    f"peak val AP={peak_val_ap:.4f}@{peak_val_epoch}"
                    if peak_val_epoch is not None and peak_val_ap is not None
                    else ""
                )
                + (
                    f" | best test AP={parsed.best_test_ap:.4f}"
                    if parsed.best_test_ap is not None
                    else ""
                ),
            ]
            if part
        )
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Val/Test AP curves for each seed log in a sweep directory.")
    parser.add_argument("log_dir", type=Path, help="Directory containing seed_*.log files.")
    parser.add_argument("--pattern", default="seed_*.log", help="Glob pattern used to find logs.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for output PNG files. Defaults to <log_dir>/plots.")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else log_dir / "plots"

    if not log_dir.exists():
        raise SystemExit(f"log_dir does not exist: {log_dir}")

    log_files = sorted(path for path in log_dir.glob(args.pattern) if path.is_file())
    if not log_files:
        raise SystemExit(f"No log files matched {args.pattern} in {log_dir}")

    for log_file in log_files:
        parsed = parse_log(log_file)
        output_path = output_dir / f"{log_file.stem}_ap_curve.png"
        plot_log(parsed, output_path, dpi=args.dpi)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
