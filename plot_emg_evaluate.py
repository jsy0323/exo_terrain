import csv
import os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from Env.MFG_Musculoskelet_V11.mfgenv.ReferTraj_V8 import TrajectoryManager
from Env.MFG_Musculoskelet_V11.mfgenv.mfg_MSenv import MFG_Musculoskeletal_V11

def build_env_for_actuator_names():
    # 读取默认环境配置
    with open("Env/MFG_Musculoskelet_V11/cfg/mujocoenv_default.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # 构建 TrajectoryManager（只为成功初始化 env）
    base_cfg = dict(base_cfg)
    # 新版 MuJoCoEnv 构造时要求带上 default_frame_skip。
    # 这里先给一个安全占位值，reset 后环境会按轨迹重新计算实际 frame_skip。
    base_cfg.setdefault("default_frame_skip", 1)

    tm_args = base_cfg["traj_manager"]
    shared_tm = TrajectoryManager(
        data_path=tm_args["data_path"],
        sample_frequency=float(tm_args["sample_frequency"]),
        speed_range=(float(tm_args["speed_range"][0]), float(tm_args["speed_range"][1])),
        verbose=bool(tm_args["verbose"]),
    )
    base_cfg["traj_manager"] = shared_tm

    return MFG_Musculoskeletal_V11(config=base_cfg, render_mode=None)

def get_actuator_names(base_env):
    """
    尽量兼容不同 mujoco python 包装：
    - 如果有 base_env.model.actuator(i).name
    - 或有 base_env.model.actuator_names
    """
    model = base_env.model

    names = None

    # 1) 常见：model.actuator_names
    if hasattr(model, "actuator_names"):
        try:
            names = list(model.actuator_names)
            names = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in names]
        except Exception:
            names = None

    # 2) 另一种：model.actuator(i).name
    if names is None and hasattr(model, "nu"):
        try:
            names = []
            for i in range(model.nu):
                a = model.actuator(i)
                n = a.name
                n = n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                names.append(n)
        except Exception:
            names = None

    if names is None:
        raise RuntimeError("Could not fetch actuator names from env.model. Please check your mujoco bindings.")
    return names

def get_activation_names(base_env):
    """
    返回一个长度为 model.na 的列表：act_names[act_idx] = 对应 actuator 名称
    """
    model = base_env.model
    nu = int(model.nu)
    na = int(model.na)

    # actuator 名称
    if hasattr(model, "actuator_names"):
        actuator_names = [
            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
            for n in list(model.actuator_names)
        ]
    else:
        actuator_names = []
        for i in range(nu):
            n = model.actuator(i).name
            actuator_names.append(n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n))

    # 关键：actuator -> act index
    actadr = np.asarray(model.actuator_actadr).astype(int)  # shape (nu,)

    # 有些 MuJoCo 版本有 actuator_actnum（每个 actuator 占用多少 act state）
    if hasattr(model, "actuator_actnum"):
        actnum = np.asarray(model.actuator_actnum).astype(int)  # shape (nu,)
    else:
        actnum = np.ones(nu, dtype=int)

    act_names = [""] * na
    for i in range(nu):
        adr = actadr[i]
        if adr < 0:
            continue
        k = int(actnum[i])
        # 多维 activation 的情况：用 name#0, name#1...
        for t in range(k):
            idx = adr + t
            if 0 <= idx < na:
                act_names[idx] = actuator_names[i] if k == 1 else f"{actuator_names[i]}#{t}"

    return act_names


def pick_thigh_act_indices(act_names):
    keys = [
        "rect", "rf", "vast", "vl", "vm", "vi",     # quads
        "bifem", "bf", "semit", "st", "semim", "sm",# hams
        "glut", "glute",                            # glute
        "add", "adduc",                             # adductors
        "iliop", "psoa", "psoas", "iliac",          # hip flexors (optional)
    ]
    idx, names = [], []
    for i, n in enumerate(act_names):
        ln = (n or "").lower()
        if ln and any(k in ln for k in keys):
            idx.append(i)
            names.append(n)
    return idx, names

from scipy.signal import butter, filtfilt
from scipy.interpolate import PchipInterpolator


CASE_SPECS = [
    {
        "npz_path": Path("logs/exo_terrain_at/traj_emg_best1/traj_0.npz"),
        "label": "Cable Exoskeleton",
        "color": "#0072B2",
    },
    # {
    #     "npz_path": Path("models/exo_vir/traj_emg_best/traj_1.npz"),
    #     "label": "Virtual Exoskeleton",
    #     "color": "#D55E00",
    # },
    # {
    #     "npz_path": Path("models/exo_obs_tes/traj_emg_best/traj_1.npz"),
    #     "label": "Cable Exoskeleton",
    #     "color": "#E69F00",
    # },
    {
        "npz_path": Path("logs/exo_terrain_at/traj_emg_best/traj_0.npz"),
        "label": "No Exoskeleton",
        "color": "#4D4D4D",
    },
]

FIG_WIDTH_INCH = 4.8
ROW_HEIGHT_INCH = 1.9
OUTPUT_DIR = Path("plots_exo_terrain")
ACTIVATION_SMOOTHING_WINDOW = 7
EXO_OUTPUT_SMOOTHING_WINDOW = 7
EXO_FORCE_SCALE_N = 200.0
EXO_FORCE_YLIM_N = (0.0, 200.0)
EXO_VIR_TORQUE_SCALE_NM = 50.0
EXO_VIR_TORQUE_SIGN = -1.0
EXO_SIDES_TO_PLOT = {"left"}  # Use {"left", "right"} to show both motors.
REQUIRE_MATCHING_TERRAIN = True


def configure_publication_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def format_muscle_label(name):
    return name.replace("_", " ")


def format_motor_label(name):
    return str(name).replace("_", " ")


def style_axis(ax, show_xlabel=False):
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    if not show_xlabel:
        ax.tick_params(labelbottom=False)


def lowpass_butter(x, fs, cutoff=6.0, order=4):
    # cutoff: Hz，常用 4~8Hz 做 EMG envelope
    nyq = 0.5 * fs
    wn = cutoff / nyq
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)

def moving_average(x, win=15):
    win = int(win)
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, kernel, mode="same")


def preprocess_activation_signal(x, smooth_window=ACTIVATION_SMOOTHING_WINDOW):
    """Keep the plotted signal consistent with muscle activation semantics."""
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, 1.0)
    if smooth_window and smooth_window > 1:
        x = moving_average(x, win=smooth_window)
    return np.clip(x, 0.0, 1.0)


def preprocess_exo_signal(x, smooth_window=EXO_OUTPUT_SMOOTHING_WINDOW):
    """Smooth applied exoskeleton control before gait-phase resampling."""
    x = np.asarray(x, dtype=np.float64)
    if smooth_window and smooth_window > 1:
        x = moving_average(x, win=smooth_window)
    return x


def summarize_curve(y):
    y = np.asarray(y, dtype=np.float64)
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "max": float(np.max(y)),
    }


def resample_by_phase_pchip(phase_pct, y, n_grid=1001, cycle_idx=0, wrap_threshold=-50):
    """稳健：回绕切段 + 去重 + PCHIP 插值到密集 phase 网格

    cycle_idx:
        0 表示第一个步态周期（默认）
        1 表示跳过第一个周期，使用第二个周期
    wrap_threshold:
        相位从接近100跳回接近0的判定阈值（d < wrap_threshold）
    """
    phase_pct = np.asarray(phase_pct, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # 1) 回绕检测：从接近100跳回接近0（阈值可调）
    d = np.diff(phase_pct)
    cut = np.where(d < wrap_threshold)[0]
    if len(cut) > 0:
        if cycle_idx <= 0:
            start, end = 0, cut[0] + 1
        else:
            if len(cut) >= cycle_idx:
                start = cut[cycle_idx - 1] + 1
                end = cut[cycle_idx] + 1 if len(cut) > cycle_idx else len(phase_pct)
            else:
                # 不足的周期时回退到第一个周期
                start, end = 0, cut[0] + 1
        phase_pct = phase_pct[start:end]
        y = y[start:end]

    # 2) 按 phase 排序（只对这一段，不会把回绕拼起来）
    order = np.argsort(phase_pct)
    x = phase_pct[order]
    yy = y[order]

    # 3) 去掉重复 phase（PCHIP 需要 x 严格递增）
    x_unique, idx_unique = np.unique(x, return_index=True)
    y_unique = yy[idx_unique]
    if len(x_unique) < 2:
        raise ValueError(f"Not enough unique phase points: {len(x_unique)}")

    grid = np.linspace(x_unique[0], x_unique[-1], n_grid)
    f = PchipInterpolator(x_unique, y_unique, extrapolate=True)
    return grid, f(grid)


def to_phase_pct(phase):
    return phase * 100.0 if np.max(phase) <= 1.5 else phase


def load_motor_names(npz_data, motor_dim):
    if "motor_names" not in npz_data.files:
        return [f"motor_{i}" for i in range(motor_dim)]

    names = []
    for name in npz_data["motor_names"]:
        if isinstance(name, (bytes, bytearray)):
            names.append(name.decode("utf-8"))
        else:
            names.append(str(name))
    if len(names) != motor_dim:
        return [f"motor_{i}" for i in range(motor_dim)]
    return names


def load_exo_names(npz_data, exo_dim):
    if "exo_names" not in npz_data.files:
        return [f"exo_{i}" for i in range(exo_dim)]

    names = []
    for name in npz_data["exo_names"]:
        if isinstance(name, (bytes, bytearray)):
            names.append(name.decode("utf-8"))
        else:
            names.append(str(name))
    if len(names) != exo_dim:
        return [f"exo_{i}" for i in range(exo_dim)]
    return names


def pick_exo_motor_r_index(exo_names):
    for i, name in enumerate(exo_names):
        if str(name).lower() == "ankle_motor_r":
            return i, str(name)
    for i, name in enumerate(exo_names):
        if str(name).lower().endswith("_r"):
            return i, str(name)
    return None, None


def is_virtual_exo_case(spec):
    label = str(spec.get("label", "")).lower()
    npz_path = str(spec.get("npz_path", "")).lower()
    return ("virtual" in label) or ("exo_vir" in npz_path) or ("_vir" in npz_path)


def is_virtual_exo_output(spec, actuator_name):
    """Return True when an exo channel represents joint torque, not cable force."""
    name = str(actuator_name).lower()
    return is_virtual_exo_case(spec) or name.startswith("exo_ankle_")


def exo_side_style(actuator_name, actuator_index):
    """Return a readable side label and line style for an exo actuator."""
    name = str(actuator_name).lower()
    if name.endswith("_r"):
        return "right", "-"
    if name.endswith("_l"):
        return "left", "--"
    return format_motor_label(actuator_name), ("-" if actuator_index == 0 else "--")


def validate_matching_terrain(case_data):
    missing = [str(case["npz_path"]) for case in case_data if not case["terrain_hash"]]
    if missing:
        message = (
            "Cannot verify a same-terrain comparison because these NPZ files have no "
            f"terrain_hash: {missing}. Re-export them with export_traj_emg.py."
        )
        if REQUIRE_MATCHING_TERRAIN:
            raise ValueError(message)
        print(f"WARNING: {message}")
        return

    hashes = {case["terrain_hash"] for case in case_data}
    if len(hashes) != 1:
        details = ", ".join(
            f"{case['label']}={case['terrain_hash'][:12]} "
            f"(seed={case['terrain_seed']})"
            for case in case_data
        )
        raise ValueError(f"Terrain mismatch between comparison cases: {details}")

    terrain_hash = next(iter(hashes))
    print(
        f"Verified matching terrain for {len(case_data)} cases: "
        f"hash={terrain_hash[:12]}"
    )


def main():
    configure_publication_style()

    case_data = []
    for spec in CASE_SPECS:
        with np.load(spec["npz_path"], allow_pickle=True) as data:
            case_entry = {
                **spec,
                "phase_pct": to_phase_pct(data["phase"]),
                "emg": np.asarray(data["emg"], dtype=np.float32),
                "motor_ctrl": None,
                "motor_names": [],
                "exo_ctrl": None,
                "exo_names": [],
                "terrain_hash": (
                    str(data["terrain_hash"].item())
                    if "terrain_hash" in data.files
                    else ""
                ),
                "terrain_seed": (
                    int(data["terrain_seed"].item())
                    if "terrain_seed" in data.files
                    else None
                ),
            }
            if "motor_ctrl" in data.files:
                motor_ctrl = np.asarray(data["motor_ctrl"], dtype=np.float32)
                if motor_ctrl.ndim == 1:
                    motor_ctrl = motor_ctrl[:, np.newaxis]
                case_entry["motor_ctrl"] = motor_ctrl
                case_entry["motor_names"] = load_motor_names(data, motor_ctrl.shape[1])
            if "exo_ctrl" in data.files:
                exo_ctrl = np.asarray(data["exo_ctrl"], dtype=np.float32)
                if exo_ctrl.ndim == 1:
                    exo_ctrl = exo_ctrl[:, np.newaxis]
                case_entry["exo_ctrl"] = exo_ctrl
                case_entry["exo_names"] = load_exo_names(data, exo_ctrl.shape[1])
            case_data.append(case_entry)

    validate_matching_terrain(case_data)

    base_env = build_env_for_actuator_names()
    try:
        act_names = get_activation_names(base_env)  # len == model.na == 80

        muscle_names = ["soleus_l"]
        # muscle_names = ["glmax1_r", "glmax2_r", "glmax3_r"]

        col_idx = []
        for n in muscle_names:
            if n not in act_names:
                raise RuntimeError(f"{n} not found in activation names.")
            col_idx.append(act_names.index(n))

        cycle_idx = 1  # 0=第一个周期；1=跳过第一个周期使用第二个周期

        n_rows = len(muscle_names)
        total_rows = n_rows  # 修改 1：不再加 1，总行数等于肌肉数量
        fig_height = ROW_HEIGHT_INCH * total_rows + 0.7
        has_exo_output = any(
            spec["exo_ctrl"] is not None and spec["exo_ctrl"].size > 0
            for spec in case_data
        )
        total_rows = n_rows + int(has_exo_output)
        fig_height = ROW_HEIGHT_INCH * total_rows + 0.7
        fig, axes = plt.subplots(
            total_rows,
            1,
            figsize=(FIG_WIDTH_INCH, fig_height),
            dpi=300,
            sharex=True,
            squeeze=False,
        )
        axes = axes.flatten()
        
        muscle_axes = axes  # 修改 2：现在所有的 axes 都分配给肌肉
        # motor_ax = axes[-1]           <-- 删除了这些
        # motor_ax_right = motor_ax.twinx() <-- 删除了这些
        
        muscle_axes = axes[:n_rows]
        exo_ax = axes[-1] if has_exo_output else None
        stats_rows = []
        exo_stats_rows = []

        legend_handles = None
        for row, (ax, j, muscle_name) in enumerate(zip(muscle_axes, col_idx, muscle_names)):
            y_max = 0.0
            for spec in case_data:
                signal = spec["emg"][:, j]
                signal_processed = preprocess_activation_signal(signal)
                grid, signal_resampled = resample_by_phase_pchip(
                    spec["phase_pct"],
                    signal_processed,
                    n_grid=1001,
                    cycle_idx=cycle_idx,
                )
                signal_resampled = np.clip(signal_resampled, 0.0, 1.0)
                curve_stats = summarize_curve(signal_resampled)
                stats_rows.append(
                    {
                        "muscle": muscle_name,
                        "case": spec["label"],
                        "mean": curve_stats["mean"],
                        "std": curve_stats["std"],
                        "max": curve_stats["max"],
                    }
                )
                handle = ax.plot(
                    grid,
                    signal_resampled,
                    color=spec["color"],
                    label=spec["label"],
                )[0]
                y_max = max(y_max, float(np.max(signal_resampled)))
                if legend_handles is None:
                    legend_handles = []
                if row == 0:
                    legend_handles.append(handle)

            ax.set_ylim(0.0, y_max * 1.08 if y_max > 0 else 1.0)
            
            # 修改 3：让最后一行肌肉图显示 X 轴刻度，而不是全隐藏
            is_last_row = (row == n_rows - 1 and exo_ax is None)
            style_axis(ax, show_xlabel=is_last_row) 
            
            if row == 0:
                ax.set_ylabel("Muscle activation")
            ax.text(
                0.02,
                0.92,
                format_muscle_label(muscle_name),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
            )

        if exo_ax is not None:
            force_curves = []
            torque_curves = []
            for spec in case_data:
                exo_ctrl = spec["exo_ctrl"]
                if exo_ctrl is None or exo_ctrl.size == 0:
                    continue

                for actuator_index, actuator_name in enumerate(spec["exo_names"]):
                    exo_signal = preprocess_exo_signal(
                        exo_ctrl[:, actuator_index]
                    )
                    grid, output = resample_by_phase_pchip(
                        spec["phase_pct"],
                        exo_signal,
                        n_grid=1001,
                        cycle_idx=cycle_idx,
                    )
                    side, line_style = exo_side_style(actuator_name, actuator_index)
                    if side not in EXO_SIDES_TO_PLOT:
                        continue
                    label = f"{spec['label']} ({side})"

                    if is_virtual_exo_output(spec, actuator_name):
                        output = (
                            np.asarray(output, dtype=np.float64)
                            * EXO_VIR_TORQUE_SCALE_NM
                            * EXO_VIR_TORQUE_SIGN
                        )
                        output_type = "torque_nm"
                        torque_curves.append(
                            (grid, output, spec["color"], line_style, label)
                        )
                    else:
                        output = (
                            np.clip(np.asarray(output, dtype=np.float64), 0.0, 1.0)
                            * EXO_FORCE_SCALE_N
                        )
                        output_type = "force_n"
                        force_curves.append(
                            (grid, output, spec["color"], line_style, label)
                        )

                    curve_stats = summarize_curve(output)
                    exo_stats_rows.append(
                        {
                            "actuator": actuator_name,
                            "case": spec["label"],
                            "output_type": output_type,
                            "mean": curve_stats["mean"],
                            "std": curve_stats["std"],
                            "max": curve_stats["max"],
                            "min": float(np.min(output)),
                        }
                    )

            style_axis(exo_ax, show_xlabel=True)
            exo_ax.text(
                0.02,
                0.92,
                "Exoskeleton output",
                transform=exo_ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
            )

            exo_legend_handles = []
            for grid, output, color, line_style, label in force_curves:
                handle = exo_ax.plot(
                    grid,
                    output,
                    color=color,
                    linestyle=line_style,
                    linewidth=1.8,
                    label=label,
                )[0]
                exo_legend_handles.append(handle)
            if force_curves:
                exo_ax.set_ylabel("Cable force\n(N)")
                exo_ax.set_ylim(*EXO_FORCE_YLIM_N)

            if torque_curves:
                torque_ax = exo_ax.twinx() if force_curves else exo_ax
                torque_min = np.inf
                torque_max = -np.inf
                for grid, output, color, line_style, label in torque_curves:
                    handle = torque_ax.plot(
                        grid,
                        output,
                        color=color,
                        linestyle=line_style,
                        linewidth=1.8,
                        label=label,
                    )[0]
                    exo_legend_handles.append(handle)
                    torque_min = min(torque_min, float(np.min(output)))
                    torque_max = max(torque_max, float(np.max(output)))
                torque_ax.set_ylabel("Exoskeleton torque\n(Nm)")
                torque_span = torque_max - torque_min
                torque_pad = (
                    0.08 * torque_span
                    if torque_span > 1.0e-6
                    else max(0.05 * max(abs(torque_min), abs(torque_max), 1.0), 0.05)
                )
                torque_ax.set_ylim(torque_min - torque_pad, torque_max + torque_pad)
                if torque_ax is not exo_ax:
                    torque_ax.spines["top"].set_visible(False)
                    torque_ax.grid(False)
                    torque_ax.tick_params(direction="out", length=3.5, width=0.8)

            if exo_legend_handles:
                exo_ax.legend(
                    handles=exo_legend_handles,
                    loc="upper right",
                    frameon=False,
                    fontsize=8,
                    handlelength=2.4,
                )

        # 修改 4：在这里删除了原来所有的 motor 绘图逻辑 
        # (包含 right_motor_y_min, 循环遍历 exo_ctrl, style_axis(motor_ax...) 等全删掉)

        fig.supxlabel("Gait cycle (%)", y=0.03) # 整体 X 轴标签保留

        # n_rows = len(muscle_names)
        # total_rows = n_rows + 1
        # fig_height = ROW_HEIGHT_INCH * total_rows + 0.7
        # fig, axes = plt.subplots(
        #     total_rows,
        #     1,
        #     figsize=(FIG_WIDTH_INCH, fig_height),
        #     dpi=300,
        #     sharex=True,
        #     squeeze=False,
        # )
        # axes = axes.flatten()
        # muscle_axes = axes[:n_rows]
        # motor_ax = axes[-1]
        # motor_ax_right = motor_ax.twinx()
        # stats_rows = []

        # legend_handles = None
        # for row, (ax, j, muscle_name) in enumerate(zip(muscle_axes, col_idx, muscle_names)):
        #     y_max = 0.0
        #     for spec in case_data:
        #         signal = spec["emg"][:, j]
        #         signal_processed = preprocess_activation_signal(signal)
        #         grid, signal_resampled = resample_by_phase_pchip(
        #             spec["phase_pct"],
        #             signal_processed,
        #             n_grid=1001,
        #             cycle_idx=cycle_idx,
        #         )
        #         signal_resampled = np.clip(signal_resampled, 0.0, 1.0)
        #         curve_stats = summarize_curve(signal_resampled)
        #         stats_rows.append(
        #             {
        #                 "muscle": muscle_name,
        #                 "case": spec["label"],
        #                 "mean": curve_stats["mean"],
        #                 "std": curve_stats["std"],
        #                 "max": curve_stats["max"],
        #             }
        #         )
        #         handle = ax.plot(
        #             grid,
        #             signal_resampled,
        #             color=spec["color"],
        #             label=spec["label"],
        #         )[0]
        #         y_max = max(y_max, float(np.max(signal_resampled)))
        #         if legend_handles is None:
        #             legend_handles = []
        #         if row == 0:
        #             legend_handles.append(handle)

        #     ax.set_ylim(0.0, y_max * 1.08 if y_max > 0 else 1.0)
        #     style_axis(ax, show_xlabel=False)
        #     if row == 0:
        #         ax.set_ylabel("Muscle activation")
        #     ax.text(
        #         0.02,
        #         0.92,
        #         format_muscle_label(muscle_name),
        #         transform=ax.transAxes,
        #         ha="left",
        #         va="top",
        #         fontsize=9,
        #         fontweight="bold",
        #     )

        # right_motor_y_min = np.inf
        # right_motor_y_max = -np.inf
        # has_force_plot = False
        # has_torque_plot = False
        # for spec in case_data:
        #     exo_ctrl = spec["exo_ctrl"]
        #     if exo_ctrl is None or exo_ctrl.size == 0:
        #         continue

        #     right_motor_idx, right_motor_name = pick_exo_motor_r_index(spec["exo_names"])
        #     if right_motor_idx is None:
        #         continue

        #     grid, motor_resampled = resample_by_phase_pchip(
        #         spec["phase_pct"],
        #         exo_ctrl[:, right_motor_idx],
        #         n_grid=1001,
        #         cycle_idx=cycle_idx,
        #     )
        #     if is_virtual_exo_case(spec):
        #         motor_resampled = (
        #             np.asarray(motor_resampled, dtype=np.float64)
        #             * EXO_VIR_TORQUE_SCALE_NM
        #             * EXO_VIR_TORQUE_SIGN
        #         )
        #         motor_ax_right.plot(
        #             grid,
        #             motor_resampled,
        #             color=spec["color"],
        #             linewidth=1.6,
        #             label=spec["label"],
        #         )
        #         right_motor_y_min = min(right_motor_y_min, float(np.min(motor_resampled)))
        #         right_motor_y_max = max(right_motor_y_max, float(np.max(motor_resampled)))
        #         has_torque_plot = True
        #     else:
        #         motor_resampled = np.clip(motor_resampled, 0.0, 1.0) * EXO_FORCE_SCALE_N
        #         motor_ax.plot(
        #             grid,
        #             motor_resampled,
        #             color=spec["color"],
        #             linewidth=1.6,
        #             label=spec["label"],
        #         )
        #         has_force_plot = True

        # style_axis(motor_ax, show_xlabel=True)
        # motor_ax.set_ylabel("Exoskeleton force\n(N)")
        # motor_ax.text(
        #     0.02,
        #     0.92,
        #     "Exoskeleton motor output",
        #     transform=motor_ax.transAxes,
        #     ha="left",
        #     va="top",
        #     fontsize=9,
        #     fontweight="bold",
        # )
        # motor_ax.set_ylim(*EXO_FORCE_YLIM_N)

        # motor_ax_right.set_xlim(0.0, 100.0)
        # motor_ax_right.spines["top"].set_visible(False)
        # motor_ax_right.grid(False)
        # motor_ax_right.tick_params(direction="out", length=3.5, width=0.8)
        # if has_torque_plot:
        #     motor_ax_right.set_ylabel("Exoskeleton torque\n(Nm)")
        #     span = right_motor_y_max - right_motor_y_min
        #     pad = 0.08 * span if span > 1.0e-6 else max(
        #         0.05 * max(abs(right_motor_y_min), abs(right_motor_y_max), 1.0),
        #         0.05,
        #     )
        #     motor_ax_right.set_ylim(right_motor_y_min - pad, right_motor_y_max + pad)
        # else:
        #     motor_ax_right.set_yticks([])
        #     motor_ax_right.spines["right"].set_visible(False)

        # fig.supxlabel("Gait cycle (%)", y=0.03)
        fig.legend(
            legend_handles,
            [spec["label"] for spec in case_data],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(case_data),
            frameon=False,
            handlelength=2.5,
            columnspacing=1.5,
        )
        fig.tight_layout(rect=(0.04, 0.05, 1.0, 0.94))

        output_name = f"{'_'.join(muscle_names)}_{CASE_SPECS[0]['npz_path'].stem}_compare_publication"
        OUTPUT_DIR.mkdir(exist_ok=True)
        stats_path = OUTPUT_DIR / f"{output_name}_stats.csv"
        with stats_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["muscle", "case", "mean", "std", "max"])
            writer.writeheader()
            for row in stats_rows:
                writer.writerow(
                    {
                        "muscle": row["muscle"],
                        "case": row["case"],
                        "mean": f"{row['mean']:.6f}",
                        "std": f"{row['std']:.6f}",
                        "max": f"{row['max']:.6f}",
                    }
                )

        print("Curve statistics:")
        for row in stats_rows:
            print(
                f"  {row['muscle']} | {row['case']}: "
                f"mean={row['mean']:.6f}, std={row['std']:.6f}, max={row['max']:.6f}"
            )
        print(f"Saved: {stats_path.resolve()}")

        if exo_stats_rows:
            exo_stats_path = OUTPUT_DIR / f"{output_name}_exo_stats.csv"
            with exo_stats_path.open("w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "actuator",
                    "case",
                    "output_type",
                    "mean",
                    "std",
                    "min",
                    "max",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in exo_stats_rows:
                    writer.writerow(
                        {
                            "actuator": row["actuator"],
                            "case": row["case"],
                            "output_type": row["output_type"],
                            "mean": f"{row['mean']:.6f}",
                            "std": f"{row['std']:.6f}",
                            "min": f"{row['min']:.6f}",
                            "max": f"{row['max']:.6f}",
                        }
                    )
            print("Exoskeleton output statistics:")
            for row in exo_stats_rows:
                print(
                    f"  {row['actuator']} | {row['case']}: "
                    f"mean={row['mean']:.3f}, std={row['std']:.3f}, "
                    f"min={row['min']:.3f}, max={row['max']:.3f} "
                    f"({row['output_type']})"
                )
            print(f"Saved: {exo_stats_path.resolve()}")

        for suffix in [".png", ".pdf", ".svg"]:
            out_path = OUTPUT_DIR / f"{output_name}{suffix}"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {out_path.resolve()}")
        plt.close(fig)
    finally:
        base_env.close()


if __name__ == "__main__":
    main()
