#!/usr/bin/env python3
"""Plot shifted mean reward CSV files with matplotlib."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


CSV_DIR = Path("csv")
DEFAULT_OUTPUT = CSV_DIR / "mean_reward_compared.png"
BASELINE_NAME = "mean_reward_baseline.csv"
SHIFT_START = 2000.0
LABEL_OVERRIDES = {
    "mean_reward_baseline": "warmup baseline",
    "mean_reward_140": "exo_140",
    "mean_reward_256": "exo_256",
    "mean_reward_469": "exo_469",
    "mean_reward_pa": "exo_pa",
    "mean_reward_re": "resume_without_exo",
    "mean_reward_vir": "exo_vir",
}
COLORS = [
    "#0f766e",
    "#d9485f",
    "#2563eb",
    "#ea580c",
    "#7c3aed",
    "#0891b2",
    "#65a30d",
]
DEFAULT_SMOOTH_WINDOW = 81


def load_mean_reward_csv(path: Path) -> tuple[str, list[tuple[float, float]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")

        step_key = find_step_key(reader.fieldnames)
        reward_key = find_reward_key(reader.fieldnames)
        rows: list[tuple[float, float]] = []
        for row in reader:
            rows.append((float(row[step_key]), float(row[reward_key])))

    if not rows:
        raise ValueError(f"{path} has no data rows.")

    return path.stem, rows


def find_step_key(fieldnames: list[str]) -> str:
    for candidate in fieldnames:
        if candidate.lower() == "step":
            return candidate
    raise ValueError(f"Step column not found in {fieldnames}.")


def find_reward_key(fieldnames: list[str]) -> str:
    for candidate in fieldnames:
        if candidate.lower() == "mean_reward":
            return candidate

    for candidate in fieldnames:
        lowered = candidate.lower()
        if "train/mean_reward" in lowered and "__min" not in lowered and "__max" not in lowered:
            return candidate

    raise ValueError(f"Mean reward column not found in {fieldnames}.")


def shift_series(path: Path, rows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    first_step = rows[0][0]
    offset = 0.0 if path.name == BASELINE_NAME else SHIFT_START
    return [(step - first_step + offset, reward) for step, reward in rows]


def discover_series(csv_dir: Path) -> list[tuple[str, list[tuple[float, float]]]]:
    paths = sorted(csv_dir.glob("mean_reward*.csv"))
    if not paths:
        raise ValueError(f"No mean_reward*.csv files found in {csv_dir}.")

    ordered_paths = sorted(paths, key=lambda path: (path.name != BASELINE_NAME, path.name))
    series = []
    for path in ordered_paths:
        label, rows = load_mean_reward_csv(path)
        series.append((label, shift_series(path, rows)))
    return series


def format_label(label: str) -> str:
    manual_label = LABEL_OVERRIDES.get(label)
    if manual_label is not None:
        return manual_label

    if label == Path(BASELINE_NAME).stem:
        return Path(BASELINE_NAME).stem
    if label.startswith("mean_reward_"):
        return label.removeprefix("mean_reward_")
    return label


def normalize_window(window: int, series_length: int) -> int:
    window = max(1, min(window, series_length))
    if window % 2 == 0:
        window = max(1, window - 1)
    return window


def smooth_values(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return values[:]

    import numpy as np

    normalized_window = normalize_window(window, len(values))
    if normalized_window <= 1:
        return values[:]

    pad = normalized_window // 2
    kernel = np.ones(normalized_window, dtype=float) / normalized_window
    padded = np.pad(np.asarray(values, dtype=float), (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.tolist()


def render_plot(
    series: list[tuple[str, list[tuple[float, float]]]],
    output_path: Path,
    smooth_window: int,
) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required. Run with '.venv/bin/python plot_shifted_mean_reward.py' "
            "or install matplotlib into your environment."
        ) from exc

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=220)
    fig.patch.set_facecolor("#f7f7f4")
    ax.set_facecolor("#ffffff")

    for index, (label, rows) in enumerate(series):
        x_values = [step for step, _ in rows]
        y_values = [reward for _, reward in rows]
        y_smooth = smooth_values(y_values, smooth_window)
        is_baseline = label == Path(BASELINE_NAME).stem
        color = COLORS[index % len(COLORS)]

        ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=0.9,
            alpha=0.14 if is_baseline else 0.10,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            x_values,
            y_smooth,
            label=format_label(label),
            color=color,
            linewidth=3.2 if is_baseline else 2.4,
            alpha=1.0 if is_baseline else 0.9,
            solid_capstyle="round",
            zorder=4 if is_baseline else 3,
        )

    ax.axvline(
        SHIFT_START,
        color="#111827",
        linestyle=(0, (6, 6)),
        linewidth=2.2,
        alpha=0.95,
        zorder=2,
    )
    ax.annotate(
        "resume @ step 2000",
        xy=(SHIFT_START, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(8, -10),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.2", "fc": "#ffffff", "ec": "#111827", "alpha": 0.92},
    )

    ax.set_title("Mean Reward Comparison", fontsize=20, weight="bold", pad=18)
    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel("Mean Reward", fontsize=14)
    ax.grid(True, which="major", color="#d9dde3", linewidth=0.9)
    ax.grid(True, which="minor", color="#edf0f2", linewidth=0.6)
    ax.minorticks_on()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa0a6")
    ax.spines["bottom"].set_color("#9aa0a6")

    legend = ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.988, 0.02),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.45,
        labelspacing=0.45,
        columnspacing=1.1,
        handlelength=2.2,
        handletextpad=0.55,
        fontsize=10,
    )
    legend.get_frame().set_edgecolor("#d0d7de")
    legend.get_frame().set_facecolor("#ffffff")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR, help="Directory containing mean_reward CSV files.")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Output path, e.g. .png or .svg.")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Odd moving-average window used to smooth each curve.",
    )
    args = parser.parse_args()

    series = discover_series(args.csv_dir)
    render_plot(series, args.output, args.smooth_window)
    print(f"Output: {args.output}")
    print(f"Smooth window: {args.smooth_window}")
    print(f"Series: {len(series)}")
    for label, rows in series:
        print(f"{label}: points={len(rows)}, x_range={rows[0][0]:.0f}..{rows[-1][0]:.0f}")


if __name__ == "__main__":
    main()
