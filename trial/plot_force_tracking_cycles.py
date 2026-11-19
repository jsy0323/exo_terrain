"""Select low-error experimental gait cycles and plot unshifted force signals.

Example: python trial/plot_force_tracking_cycles.py trial/Jin_100/trial_01
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DESIRED_COLOR = "#202938"
MEASURED_COLOR = "#0072B2"
SELECTED_COLOR = "#D55E00"


def extract_cycles(data, source, threshold, rank_by="full"):
    required = ["t_s", "desired_force", "measured_force", "valid_heel_strike_count"]
    if data[required].isna().any().any():
        raise ValueError(f"{source}: missing values in required signal columns")
    if not np.isfinite(data[required].to_numpy(dtype=float)).all():
        raise ValueError(f"{source}: non-finite signal values")
    if not np.all(np.diff(data.t_s) > 0):
        raise ValueError(f"{source}: timestamps must strictly increase")
    counts = data.valid_heel_strike_count.to_numpy()
    starts = np.flatnonzero(np.diff(counts, prepend=counts[0]) > 0)
    cycles, rows = {}, []
    # Each interval is [valid heel strike, next valid heel strike).
    # This excludes both pre-trigger samples and the final incomplete cycle.
    for start, stop in zip(starts[:-1], starts[1:]):
        cycle_id = int(counts[start])
        if counts[stop] != cycle_id + 1:
            raise ValueError(f"{source}: skipped heel-strike count after {cycle_id}")
        cycle = data.iloc[start:stop].copy()
        t0, t1 = float(data.t_s.iloc[start]), float(data.t_s.iloc[stop])
        active = cycle.desired_force.to_numpy() >= threshold
        if active.sum() < 3:
            continue
        error = (cycle.measured_force - cycle.desired_force).to_numpy()
        cycle["hs_phase_pct"] = 100 * (cycle.t_s - t0) / (t1 - t0)
        cycle["time_in_cycle_s"] = cycle.t_s - t0
        cycle["error_n"] = error
        cycle["assistance_mask"] = active.astype(int)
        cycles[cycle_id] = cycle
        rows.append({
            "source_file": source,
            "cycle_id": cycle_id,
            "start_s": t0,
            "end_s": t1,
            "duration_s": t1 - t0,
            "samples": len(cycle),
            "assistance_samples": int(active.sum()),
            "rmse_assistance_n": float(np.sqrt(np.mean(error[active] ** 2))),
            "mae_assistance_n": float(np.mean(np.abs(error[active]))),
            "rmse_full_n": float(np.sqrt(np.mean(error ** 2))),
            "mae_full_n": float(np.mean(np.abs(error))),
            "max_abs_error_full_n": float(np.max(np.abs(error))),
            "desired_peak_n": float(cycle.desired_force.max()),
            "measured_peak_n": float(cycle.measured_force.max()),
        })
    if not rows:
        raise ValueError(f"{source}: no complete cycles with assistance samples")
    metric = "rmse_full_n" if rank_by == "full" else "rmse_assistance_n"
    ranked = pd.DataFrame(rows).sort_values(
        [metric, "cycle_id"], kind="stable"
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return cycles, ranked


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e2e5e9", linewidth=0.7)
    ax.set_axisbelow(True)


def plot_signals(ax, cycle, x="hs_phase_pct"):
    ax.plot(cycle[x], cycle.desired_force, "--", color=DESIRED_COLOR,
            lw=2.0, label="Desired force")
    ax.plot(cycle[x], cycle.measured_force, color=MEASURED_COLOR,
            lw=1.8, label="Measured force")
    style_axis(ax)
    ax.set_ylabel("Force (N)")
    if x == "hs_phase_pct":
        ax.set(xlim=(0, 100), xlabel="Gait cycle (%; heel strike to heel strike)")
    else:
        ax.set_xlabel("Time within cycle (s)")


def save_figure(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(out / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def plot_record(data, source, cycles, ranked, out, top_n, threshold, rank_by):
    selected = ranked.head(top_n)
    metric = "rmse_full_n" if rank_by == "full" else "rmse_assistance_n"
    metric_label = "Full-cycle RMSE" if rank_by == "full" else "Assistance RMSE"
    method = ("Ranked by full-cycle RMSE" if rank_by == "full" else
              f"Ranked by RMSE where desired force >= {threshold:g} N")
    rows = (len(selected) + 2) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 3.4 * rows),
                             layout="constrained", squeeze=False)
    flat = axes.ravel()
    # Shared limits make differences between selected cycles easy to compare.
    measured = np.concatenate([cycles[int(i)].measured_force for i in selected.cycle_id])
    desired = np.concatenate([cycles[int(i)].desired_force for i in selected.cycle_id])
    limits = (min(-10, np.floor(measured.min() / 10) * 10),
              np.ceil(max(measured.max(), desired.max()) / 10) * 10 + 10)
    for ax, row in zip(flat, selected.itertuples()):
        cycle = cycles[row.cycle_id]
        plot_signals(ax, cycle)
        ax.set_ylim(*limits)
        ax.set_title(f"#{row.rank}  Cycle {row.cycle_id}  |  "
                     f"{row.start_s:.3f}-{row.end_s:.3f} s", loc="left", fontsize=11)
        ax.text(0.98, 0.96,
                f"Assist RMSE: {row.rmse_assistance_n:.2f} N\n"
                f"Full-cycle RMSE: {row.rmse_full_n:.2f} N",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
    flat[0].legend(loc="upper left", frameon=False, fontsize=9)
    ranking_ax = flat[len(selected)]
    chronological = ranked.sort_values("cycle_id")
    ranking_ax.bar(chronological.cycle_id, chronological[metric],
                   color=[SELECTED_COLOR if i in selected.cycle_id.values else "#b9c5cf"
                          for i in chronological.cycle_id])
    ranking_ax.set(xlabel="Cycle number", ylabel=f"{metric_label} (N)",
                   title=f"All {len(ranked)} complete cycles; selected in orange")
    ranking_ax.set_xticks(chronological.cycle_id.iloc[::2])
    style_axis(ranking_ax)
    for ax in flat[len(selected) + 1:]:
        ax.set_visible(False)
    fig.suptitle(f"Jin 100 N | {source.stem.removeprefix('raw_exo_')} | "
                 f"{len(selected)} lowest-error cycles\n"
                 f"{method}; assist metric uses desired >= {threshold:g} N\n"
                 "raw signals, no shift or smoothing", fontsize=13)
    save_figure(fig, out, "selected_cycles")

    best = selected.iloc[0]
    cycle = cycles[int(best.cycle_id)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    plot_signals(axes[0], cycle)
    plot_signals(axes[1], cycle, "time_in_cycle_s")
    for ax in axes:
        ax.set_ylim(*limits)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"{source.stem} | Best cycle {int(best.cycle_id)} | "
                 f"{best.start_s:.3f}-{best.end_s:.3f} s\n"
                 f"Assistance RMSE {best.rmse_assistance_n:.2f} N | "
                 f"Full-cycle RMSE {best.rmse_full_n:.2f} N", fontsize=12)
    save_figure(fig, out, "best_cycle")

    fig, ax = plt.subplots(figsize=(14, 4.1), layout="constrained")
    ax.plot(data.t_s, data.desired_force, "--", color=DESIRED_COLOR,
            lw=1.3, label="Desired force")
    ax.plot(data.t_s, data.measured_force, color=MEASURED_COLOR,
            lw=1.1, label="Measured force")
    for row in selected.itertuples():
        ax.axvspan(row.start_s, row.end_s, color=SELECTED_COLOR, alpha=0.13)
        ax.text((row.start_s + row.end_s) / 2, 1.01, str(row.cycle_id),
                transform=ax.get_xaxis_transform(), ha="center", fontsize=9,
                color=SELECTED_COLOR)
    ax.set(xlabel="Recorded time (s)", ylabel="Force (N)",
           xlim=(data.t_s.iloc[0], data.t_s.iloc[-1]))
    ax.legend(frameon=False, loc="upper left", ncol=2)
    style_axis(ax)
    fig.suptitle(f"{source.stem} | Selected cycle intervals shaded; cycle IDs above",
                 fontsize=12)
    save_figure(fig, out, "full_record_selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dir", type=Path, nargs="?",
                        default=Path(__file__).parent / "Jin_100" / "trial_01")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--assistance-threshold-n", type=float, default=5.0)
    parser.add_argument("--rank-by", choices=["full", "assistance"], default="full")
    args = parser.parse_args()
    if args.top_n < 1 or args.assistance_threshold_n <= 0:
        parser.error("top-n and assistance threshold must be positive")
    sources = sorted(args.trial_dir.glob("raw_exo_*.csv"))
    if not sources:
        parser.error(f"No raw_exo_*.csv found in {args.trial_dir}")
    out_root = args.trial_dir / "force_tracking_selection"
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = args.trial_dir / "trial_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    summaries, rankings = [], []
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    for source in sources:
        data = pd.read_csv(source)
        cycles, ranked = extract_cycles(data, source.name,
                                        args.assistance_threshold_n, args.rank_by)
        ranked["selected"] = ranked["rank"] <= args.top_n
        out = out_root / source.stem
        out.mkdir(exist_ok=True)
        ranked.to_csv(out / "cycle_ranking.csv", index=False)
        selected_samples = []
        for row in ranked[ranked.selected].itertuples():
            cycle = cycles[row.cycle_id].copy()
            cycle.insert(0, "cycle_id", row.cycle_id)
            cycle.insert(0, "rank", row.rank)
            selected_samples.append(cycle[[
                "rank", "cycle_id", "t_s", "time_in_cycle_s", "hs_phase_pct",
                "cycle_phase_pct", "desired_force", "measured_force", "error_n",
                "assistance_mask",
            ]])
        pd.concat(selected_samples, ignore_index=True).to_csv(
            out / "selected_cycle_samples.csv", index=False)
        plot_record(data, source, cycles, ranked, out, args.top_n,
                    args.assistance_threshold_n, args.rank_by)
        summaries.append({
            "source_file": source.name,
            "referenced_by_trial_meta": source.name == meta.get("raw_csv"),
            "complete_cycles": len(ranked),
            "selected_cycles": ranked.loc[ranked.selected, "cycle_id"].tolist(),
            "best_cycle": ranked.iloc[0].to_dict(),
        })
        rankings.append(ranked)
        print(source.name)
        print(ranked.head(args.top_n)[[
            "rank", "cycle_id", "start_s", "end_s", "rmse_assistance_n", "rmse_full_n"
        ]].round(3).to_string(index=False))
    metric = "rmse_full_n" if args.rank_by == "full" else "rmse_assistance_n"
    combined = pd.concat(rankings, ignore_index=True).sort_values(metric)
    combined.insert(0, "rank_across_files", np.arange(1, len(combined) + 1))
    combined.to_csv(out_root / "all_cycle_ranking.csv", index=False)
    summary = {
        "trial_meta_raw_csv": meta.get("raw_csv"),
        "selection_metric": "sample RMSE of measured_force - desired_force",
        "rank_by": args.rank_by,
        "assistance_threshold_n": args.assistance_threshold_n,
        "assistance_mask": "desired_force >= assistance_threshold_n",
        "cycle_definition": "[valid heel strike, next valid heel strike)",
        "plot_phase": "100 * (t - heel_strike_time) / measured_cycle_duration",
        "top_n_per_recording": args.top_n,
        "preprocessing": "none: no shift, smoothing, clipping or baseline correction",
        "records": summaries,
    }
    (out_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved plots and rankings to {out_root.resolve()}")


if __name__ == "__main__":
    main()
