# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 20:59:05 2025

@author: YAKE
"""

import os, time, pickle, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import json
import mujoco, mujoco.viewer as viewer
import quaternion, math
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# 62 dofs
keymap = [
    'pelvis_tz','pelvis_tx','pelvis_ty','pelvis_tilt','pelvis_list','pelvis_rotation',
    'hip_flexion_r','hip_adduction_r','hip_rotation_r',
    'knee_angle_r','knee_angle_r_rotation2','knee_angle_r_rotation3',
    'ankle_angle_r','subtalar_angle_r','mtp_angle_r',
    'hip_flexion_l','hip_adduction_l','hip_rotation_l',
    'knee_angle_l','knee_angle_l_rotation2','knee_angle_l_rotation3',
    'ankle_angle_l','subtalar_angle_l','mtp_angle_l',
    'lumbar_extension','lumbar_bending','lumbar_rotation',
    'thorax_extension','thorax_bending','thorax_rotation',
    'clavicle_r_flex','clavicle_r_add','clavicle_r_rot',
    'scapula_r_rotz','scapula_r_rotx','scapula_r_roty',
    'arm_flex_r','arm_add_r','arm_rot_r','elbow_flex_r','pro_sup_r','wrist_flex_r','wrist_dev_r',
    'clavicle_l_flex','clavicle_l_add','clavicle_l_rot',
    'scapula_l_rotz','scapula_l_rotx','scapula_l_roty',
    'arm_flex_l','arm_add_l','arm_rot_l','elbow_flex_l','pro_sup_l','wrist_flex_l','wrist_dev_l',
    'neck_flex','neck_add','neck_rot','head_flex','head_add','head_rot'
]
AMASS_INDEX = {name: idx for idx, name in enumerate(keymap)}

COEF = np.array([
    [-0.0108281, -0.0487847,   0.00927644,  0.0131673,   -0.00349673],  # translation2
    [ 0.0524192, -0.0150188,  -0.0340522,   0.0133393,   -0.000879151], # translation1
    [ 0.010506,   0.0247615,  -1.31647,     0.716337,    -0.138302   ], # rotation1
])

# 46 dofs
KEEP_JOINT_NAMES = [
    'pelvis_tz','pelvis_ty','pelvis_tx','pelvis_tilt','pelvis_list','pelvis_rotation',
    'hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r',
    'ankle_angle_r','subtalar_angle_r','mtp_angle_r',
    'hip_flexion_l','hip_adduction_l','hip_rotation_l','knee_angle_l',
    'ankle_angle_l','subtalar_angle_l','mtp_angle_l',
    'lumbar_extension','lumbar_bending','lumbar_rotation',
    'thorax_extension','thorax_bending','thorax_rotation',
    'arm_flex_r','arm_add_r','arm_rot_r','elbow_flex_r','pro_sup_r','wrist_flex_r','wrist_dev_r',
    'arm_flex_l','arm_add_l','arm_rot_l','elbow_flex_l','pro_sup_l','wrist_flex_l','wrist_dev_l',
    'neck_flex','neck_add','neck_rot','head_flex','head_add','head_rot'
]
KEEP_INDICES = [AMASS_INDEX[n] for n in KEEP_JOINT_NAMES]

# —— config —— 
FS = 100.0               # sampling frequency (Hz)
target_frames = 3000     # for repeat times
SPLICE_OVERLAP = 5       # overlap frames
SMOOTH_SIGMA = 3.0       # sigma
SG_WINDOW = 7            # Savitzky–Golay window size
SG_POLY = 3              # Savitzky–Golay order
MIN_DIST_S = 0.5         # minimum cycle time (s)
PROMINENCE = 0.02         


def lowpass_filter(x, fs=FS, fc=10.0, order=4):
    b, a = butter(order, fc/(fs/2), btype='low')
    return filtfilt(b, a, x, axis=0)

def remove_extra_dofs(raw_qpos: np.ndarray) -> np.ndarray:
    """from 62 to 46"""
    return raw_qpos[:, KEEP_INDICES]

_INV_SQRT2 = math.sqrt(2.0) / 2.0
_Q0_INV = quaternion.quaternion(_INV_SQRT2, -_INV_SQRT2, 0.0, 0.0)

def compute_global_quaternion(tilt, list_angle, rotation):
    for name, val in (("tilt", tilt), ("list_angle", list_angle), ("rotation", rotation)):
        if not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be a real number, got {type(val).__name__}")

    ht = tilt       * 0.5
    hl = list_angle * 0.5
    hr = rotation   * 0.5

    cz, sz = math.cos(ht), math.sin(ht)
    cx, sx = math.cos(hl), math.sin(hl)
    cy, sy = math.cos(hr), math.sin(hr)

    w1 = cz*cx
    x1 = cz*sx
    y1 = sz*sx
    z1 = sz*cx

    w2 = w1*cy - y1*sy
    x2 = x1*cy - z1*sy
    y2 = w1*sy + y1*cy
    z2 = x1*sy + z1*cy

    a  = _INV_SQRT2
    w  = a*(w2 - x2)
    x  = a*(x2 + w2)
    y  = a*(y2 - z2)
    z  = a*(z2 + y2)

    return quaternion.quaternion(w, x, y, z)

def reorient_pelvis_to_x(raw_qpos: np.ndarray) -> np.ndarray:
    assert raw_qpos.ndim == 2 and raw_qpos.shape[1] == 62
    q = raw_qpos.copy()

    iz, ix, iy = (AMASS_INDEX[k] for k in ('pelvis_tz','pelvis_tx','pelvis_ty'))
    itilt, ilist, irot = (AMASS_INDEX[k] for k in ('pelvis_tilt','pelvis_list','pelvis_rotation'))
    
    orig_tz = q[:, iz].copy()
    orig_tx = q[:, ix].copy()
    tz_centered = orig_tz - orig_tz[0]
    
    dz = tz_centered[-1]
    
    if dz >= 0:
        new_tx   =  tz_centered
        new_tz   = -orig_tx
        new_ty   =  q[:, iy].copy()
        new_list =  q[:, itilt].copy()
        new_tilt = -q[:, ilist].copy()
        new_rot  =  q[:, irot].copy() + (math.pi/2)
    else:
        new_tx   = -tz_centered
        new_tz   =  orig_tx
        new_ty   =  q[:, iy].copy()
        new_list = -q[:, itilt].copy()
        new_tilt =  q[:, ilist].copy()
        new_rot  =  q[:, irot].copy() - (math.pi/2)

    q[:, ix]    = new_tx
    q[:, iy]    = new_ty
    q[:, iz]    = new_tz
    q[:, ilist] = new_list
    q[:, itilt] = new_tilt
    q[:, irot]  = new_rot

    return q

def find_stride_minima(
        qpos46: np.ndarray,
        start_foot: str,
        fs: float = FS,
        min_dist_s: float = MIN_DIST_S,
        prom: float = PROMINENCE,
        interactive: bool = False
        ) -> np.ndarray:
    jn = 'knee_angle_l' if start_foot == 'Left' else 'knee_angle_r'
    idx = KEEP_JOINT_NAMES.index(jn)
    sig = qpos46[:, idx]
    t = np.arange(len(sig)) / fs

    strict_peaks, _ = find_peaks(
        -sig,
        distance=int(fs * min_dist_s),
        prominence=prom
    )

    if not interactive:
        print(f"[find_stride_minima] Non-interactive mode: use strict minima {strict_peaks.tolist()}")
        return strict_peaks
    
    plt.ion()
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(t, sig, label=jn)
    ax1.plot(t[strict_peaks], sig[strict_peaks],
             'r*', markersize=8, label='strict minima')
    ax1.set_title(f"{start_foot} knee minima (strict)")  
    ax1.set_xlabel("Time (s)")  
    ax1.set_ylabel("Angle (rad)")
    ax1.grid(True)
    ax1.legend()
    fig1.canvas.draw()
    fig1.show()
    plt.pause(0.1)
    plt.ioff()

    resp = input(
        f"Detected {len(strict_peaks)} strict minima at frames {strict_peaks.tolist()}.\n"
        "Press ENTER to accept, or any other key+ENTER to pick manually: "
    ).strip()
    plt.close(fig1)

    if resp == "":
        print(f"Using strict minima: {strict_peaks.tolist()}")
        return strict_peaks

    loose_peaks, _ = find_peaks(-sig)
    n_pts = len(loose_peaks)
    facecols = np.zeros((n_pts, 4))
    edgecols = np.tile([1,0,0,1], (n_pts,1))
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(t, sig, label=jn)
    pts = ax2.scatter(
        t[loose_peaks], sig[loose_peaks],
        s=80, facecolors=facecols, edgecolors=edgecols,
        picker=5, label='candidates'
    )
    ax2.set_title("click to select minima, then close")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (rad)")
    ax2.grid(True)
    ax2.legend()
    fig2.canvas.draw()
    fig2.show()
    plt.pause(0.1)

    selected = set()
    def on_pick(event):
        ind = event.ind[0]
        frame = loose_peaks[ind]
        if frame in selected:
            selected.remove(frame)
            facecols[ind] = [0,0,0,0]
            edgecols[ind] = [1,0,0,1]
        else:
            selected.add(frame)
            facecols[ind] = [0,1,0,1]
            edgecols[ind] = [0,1,0,1]
        pts.set_facecolors(facecols)
        pts.set_edgecolors(edgecols)
        fig2.canvas.draw_idle()

    fig2.canvas.mpl_connect('pick_event', on_pick)
    plt.show(block=True)
    plt.close(fig2)
    if selected:
        chosen = np.array(sorted(selected), dtype=int)
        print(f"Using user-selected minima: {chosen.tolist()}")
    else:
        chosen = strict_peaks
        print("No points selected, falling back to strict minima.")
    return np.unique(chosen)

def segment_cycles(qpos46: np.ndarray, minima: np.ndarray) -> list[np.ndarray]:
    cycles = []
    ty_idx = KEEP_JOINT_NAMES.index('pelvis_ty')
    tx_idx = KEEP_JOINT_NAMES.index('pelvis_tx')
    tz_idx = KEEP_JOINT_NAMES.index('pelvis_tz')
    for i in range(len(minima)-1):
        s, e = minima[i], minima[i+1]
        cycle = qpos46[s:e, :].copy()
        N,_ = cycle.shape

        y0, y1 = cycle[0, ty_idx], cycle[-1, ty_idx]
        trend = np.linspace(y0, y1, N)
        cycle[:, ty_idx] -= trend
        cycle[:, ty_idx] -= cycle[0, ty_idx]

        cycle[:, tx_idx] -= cycle[0, tx_idx]
        cycle[:, tz_idx] -= cycle[0, tz_idx]
        cycles.append(cycle)
    return cycles

def mirror_cycle(cycle: np.ndarray) -> np.ndarray:
    m = cycle.copy()
    idx_map = np.arange(m.shape[1])
    for i, name in enumerate(KEEP_JOINT_NAMES):
        if name.endswith('_r'):
            counterpart = name[:-2] + '_l'
            if counterpart in KEEP_JOINT_NAMES:
                j = KEEP_JOINT_NAMES.index(counterpart)
                idx_map[i], idx_map[j] = idx_map[j], idx_map[i]
    m = m[:, idx_map]
    for fn in ['pelvis_tz','pelvis_list','pelvis_rotation',
               'lumbar_bending','lumbar_rotation','thorax_bending','thorax_rotation',
               'neck_add','neck_rot','head_add','head_rot']:
        if fn in KEEP_JOINT_NAMES:
            m[:, KEEP_JOINT_NAMES.index(fn)] *= -1
    return m

def extend_and_smooth_46(
        base: np.ndarray,
        repeat: int = 10,
        splice_overlap: int = SPLICE_OVERLAP,
        smoothing_sigma: float = SMOOTH_SIGMA,
        apply_global_smoothing: bool = True,
        sg_window: int = SG_WINDOW,
        sg_poly: int = SG_POLY
        ) -> np.ndarray:

    T, D = base.shape
    assert D == len(KEEP_JOINT_NAMES)

    delta = np.zeros(D)
    for key in ('pelvis_tz','pelvis_tx','pelvis_ty'):
        idx = KEEP_JOINT_NAMES.index(key)
        delta[idx] = base[-1, idx] - base[0, idx]

    seq = np.zeros((T*repeat, D), dtype=base.dtype)
    for k in range(repeat):
        seq[k*T:(k+1)*T] = base + (delta * k)[None, :]

    total = T * repeat
    mask = np.zeros(total, dtype=bool)
    for c in range(1, repeat):
        b = c * T
        a = max(0, b - splice_overlap)
        d = min(total, b + splice_overlap)
        mask[a:d] = True

    sm = gaussian_filter1d(seq, sigma=smoothing_sigma, axis=0, mode='nearest')
    seq[mask] = sm[mask]

    if apply_global_smoothing:
        L = seq.shape[0]
        win = sg_window if sg_window % 2 == 1 else sg_window + 1
        win = min(win, L if L % 2 == 1 else L - 1)
        seq = savgol_filter(seq, win, sg_poly, axis=0)

    return seq

def compute_velocity(qpos: np.ndarray, dt: float = 1.0/FS) -> np.ndarray:
    """ act on qpos46"""
    T, D = qpos.shape
    qvel = np.zeros_like(qpos)
    if T < 2:
        return qvel
    qvel[0] = (qpos[1] - qpos[0]) / dt
    for i in range(1, T-1):
        qvel[i] = (qpos[i+1] - qpos[i-1]) / (2*dt)
    qvel[-1] = (qpos[-1] - qpos[-2]) / dt
    return qvel

def compute_beta(x: np.ndarray, xdot: np.ndarray):
    P = COEF.shape[1]
    X = np.stack([x**i for i in range(P)], axis=0)         
    angles = (COEF @ X).T                                   

    dX = np.stack([(i * x**(i-1) if i>0 else np.zeros_like(x))
                   for i in range(P)], axis=0)            
    vels = ((COEF @ dX).T) * xdot[:, None]             
    return angles, vels

def expand_beta(qpos46: np.ndarray, qvel46: np.ndarray):
    T = qpos46.shape[0]
    i_r_knee = KEEP_JOINT_NAMES.index('knee_angle_r')
    i_l_knee = KEEP_JOINT_NAMES.index('knee_angle_l')
    i_r_mtp  = KEEP_JOINT_NAMES.index('mtp_angle_r')
    i_l_mtp  = KEEP_JOINT_NAMES.index('mtp_angle_l')

    kr    = qpos46[:, i_r_knee]
    kl    = qpos46[:, i_l_knee]
    kdr   = qvel46[:, i_r_knee]
    kdl   = qvel46[:, i_l_knee]

    br_ang, br_vel = compute_beta(kr, kdr)
    bl_ang, bl_vel = compute_beta(kl, kdl)

    qpos52 = np.zeros((T, 52), dtype=qpos46.dtype)
    qvel52 = np.zeros((T, 52), dtype=qvel46.dtype)
    A_end = i_r_mtp + 1
    qpos52[:, :A_end] = qpos46[:, :A_end]
    qvel52[:, :A_end] = qvel46[:, :A_end]
    B_start = A_end
    qpos52[:, B_start:B_start+3] = br_ang
    qvel52[:, B_start:B_start+3] = br_vel
    C_start_src  = A_end
    C_end_src    = i_l_mtp + 1
    C_start_dest = B_start + 3
    qpos52[:, C_start_dest:C_start_dest+(C_end_src-C_start_src)] = qpos46[:, C_start_src:C_end_src]
    qvel52[:, C_start_dest:C_start_dest+(C_end_src-C_start_src)] = qvel46[:, C_start_src:C_end_src]
    D_start = C_start_dest + (C_end_src - C_start_src)
    qpos52[:, D_start:D_start+3] = bl_ang
    qvel52[:, D_start:D_start+3] = bl_vel
    E_start_src  = C_end_src
    E_start_dest = D_start + 3
    qpos52[:, E_start_dest:] = qpos46[:, E_start_src:]
    qvel52[:, E_start_dest:] = qvel46[:, E_start_src:]
    return qpos52, qvel52

def expand_beta_only_qpos(qpos46: np.ndarray) -> np.ndarray:
    T = qpos46.shape[0]
    kr = qpos46[:, KEEP_JOINT_NAMES.index('knee_angle_r')]
    kl = qpos46[:, KEEP_JOINT_NAMES.index('knee_angle_l')]
    
    def eval_beta(coefs, x):
        X = np.stack([x**i for i in range(coefs.shape[1])], axis=-1)
        return X @ coefs.T
    br = eval_beta(COEF, kr)  # (T,3)  [translation2, translation1, rotation1]
    bl = eval_beta(COEF, kl)  # (T,3)

    i_r = KEEP_JOINT_NAMES.index('mtp_angle_r')
    i_l = KEEP_JOINT_NAMES.index('mtp_angle_l')
    
    qpos52 = np.zeros((T, 43), dtype=qpos46.dtype)
    qpos52[:, :i_r+1] = qpos46[:, :i_r+1]
    qpos52[:, i_r+1 : i_r+1+3] = br
    qpos52[:, i_r+1+3 : i_r+1+3+(i_l-(i_r+1)+1)] = qpos46[:, i_r+1 : i_l+1]
    new_i_l = (i_l + 1) + 3
    qpos52[:, new_i_l : new_i_l+3] = bl
    tail_src = i_l+1
    tail_dest = new_i_l+3
    qpos52[:, tail_dest:] = qpos46[:, tail_src:]

    return qpos52

def _convert_single(frame52: np.ndarray) -> np.ndarray:

    tz = frame52[KEEP_JOINT_NAMES.index('pelvis_tz')]
    ty = frame52[KEEP_JOINT_NAMES.index('pelvis_ty')]
    tx = frame52[KEEP_JOINT_NAMES.index('pelvis_tx')]
    trans = np.array([tx, -tz, ty + 0.95], dtype=float)

    tilt = frame52[KEEP_JOINT_NAMES.index('pelvis_tilt')]
    lst  = frame52[KEEP_JOINT_NAMES.index('pelvis_list')]
    rot  = frame52[KEEP_JOINT_NAMES.index('pelvis_rotation')]
    qg = compute_global_quaternion(tilt, lst, rot)
    quat = np.array([qg.w, qg.x, qg.y, qg.z], dtype=np.float64)

    base53 = np.concatenate([trans, quat, frame52[6:]])
    return base53

def convert_qpos52_to_mj53(qpos52: np.ndarray) -> np.ndarray:
    arr = np.asarray(qpos52)
    if arr.ndim == 1:
        if arr.shape[0] != 52:
            raise ValueError(f"single frame must be 52 but received {arr.shape[0]}")
        return _convert_single(arr)

    if arr.ndim == 2:
        N, D = arr.shape
        if D != 52:
            raise ValueError(f"receive {D}")
        out = np.zeros((N, 53), dtype=np.float64)
        for i in range(N):
            out[i] = _convert_single(arr[i])
        return out

    raise ValueError(f"ndim={arr.ndim}")

class MuJoCoVisualizer:
    def __init__(self, model_xml: str):
        self.model  = mujoco.MjModel.from_xml_path(model_xml)
        self.data   = mujoco.MjData(self.model)
        self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.fixedcamid = -1
        self.viewer.cam.distance  = 5      
        self.viewer.cam.azimuth   = 135.0
        self.viewer.cam.elevation = -20.0
        
    def show_frame(self, qpos_frame: np.ndarray, pause: float = 1.0/10):
        self.data.qpos = qpos_frame
        mujoco.mj_fwdPosition(self.model, self.data)
        self.viewer.sync()
        time.sleep(pause)
    
    def animate_sequence(self, qpos_seq: np.ndarray, interval: float = 0.1):
        for frame in qpos_seq:
            self.show_frame(frame, pause=interval)

def extract_motion_class(file_path: str) -> str:
    stem = Path(file_path).stem
    toks = re.split(r'[_\-]', stem)

    for i in range(len(toks) - 1):
        w, s = toks[i].lower(), toks[i+1].lower()
        if w in {'walk', 'run'} and s in {'slow', 'comfortable', 'fast'}:
            return f"{w.title()}_{s.title()}"

    for tok in toks:
        lt = tok.lower()
        if lt in {'walk', 'run'}:
            return lt.title()

    return stem

def process_one_file(file_path):
    data = pickle.load(open(file_path, 'rb'))
    raw   = data['mj_qpos']
    
    aligned = reorient_pelvis_to_x(raw)
    filt    = lowpass_filter(aligned)
    q46     = remove_extra_dofs(filt)
    
    i_tx = KEEP_JOINT_NAMES.index('pelvis_tx')
    disp  = q46[-1, i_tx] - q46[0, i_tx]
    duration = q46.shape[0] / FS
    speed = abs(disp) / duration
    min_dist_s = 0.5 if speed >= 1.3 else 1.0
    
    start_foot = (
        'Left'  if q46[0, KEEP_JOINT_NAMES.index('hip_flexion_l')]
        > q46[0, KEEP_JOINT_NAMES.index('hip_flexion_r')]
        else 'Right'
    )
    
    minima = find_stride_minima(q46, start_foot, min_dist_s=min_dist_s, interactive=True)
    cycles = segment_cycles(q46, minima)
    
    motion_class = extract_motion_class(str(file_path))
    
    print(f"File: {file_path.name}")
    print(f"  Motion class : {motion_class}")
    print(f"  Speed        : {speed:.2f} m/s")
    print(f"  Start foot   : {start_foot}")
    print(f"  Segmented into {len(cycles)} cycles:")
    for idx, cyc in enumerate(cycles, start=1):
        print(f"    Cycle {idx:02d}: {cyc.shape[0]} frames")
    print("-" * 40)
    
    processed = []
    for cyc in cycles:
        Ti = cyc.shape[0]
        repeat = max(1, int(math.ceil(target_frames / Ti)))

        qpos46 = extend_and_smooth_46(cyc, repeat=repeat)
        qvel46 = compute_velocity(qpos46)
        qpos52, qvel52 = expand_beta(qpos46, qvel46)
        processed.append((qpos52.T, qvel52.T))

        cyc_m = mirror_cycle(cyc)
        qpos46_m = extend_and_smooth_46(cyc_m, repeat=repeat)
        qvel46_m = compute_velocity(qpos46_m)
        qpos52_m, qvel52_m = expand_beta(qpos46_m, qvel46_m)
        processed.append((qpos52_m.T, qvel52_m.T))
    return processed

def batch_process_trajs(input_dir, output_dir):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = []
    counts = defaultdict(int)

    for pkl_path in sorted(input_dir.glob("*.pkl")):
        motion_class = extract_motion_class(str(pkl_path))
        
        segments = process_one_file(pkl_path)

        for qpos52_T, qvel52_T in segments:
            counts[motion_class] += 1
            idx = counts[motion_class]

            fname = f"{motion_class}_{idx:03d}.npz"
            outp  = output_dir / fname
            np.savez_compressed(outp, qpos=qpos52_T, qvel=qvel52_T)

            index.append({
                "file":   fname,
                "frames": int(qpos52_T.shape[1])
            })

    with open(output_dir/"index.json", "w", encoding="utf-8") as jf:
        json.dump(index, jf, indent=2, ensure_ascii=False)

    total = sum(counts.values())
    print(f"Processed {total} segments for {len(counts)} motion classes → {output_dir}")


# XML_PATH = 'C:/Users/YAKEz/OneDrive/Desktop/MFG_Raj2015_V10.xml'
# viz = MuJoCoVisualizer(XML_PATH)

# file_path = 'C:/Users/YAKEz/OneDrive/Desktop/MFG_Musculoskelet_V10/ik_data/louis/BD004_Session2_Overground_Walk_Walk_Comfortable_Post_Process_Trial05.pkl'
# with open(file_path, 'rb') as f:
#     raw_data = pickle.load(f)['mj_qpos']
# reori_data = reorient_pelvis_to_x(raw_data)
# filtered_data = lowpass_filter(reori_data)
# filtered_46 = remove_extra_dofs(filtered_data)
# start_foot = 'Left' if filtered_46[0, KEEP_JOINT_NAMES.index('hip_flexion_l')] > filtered_46[0, KEEP_JOINT_NAMES.index('hip_flexion_r')] else 'Right'
# minima = find_stride_minima(filtered_46, start_foot, interactive=True)
# cycles = segment_cycles(filtered_46,minima)

# qpos46 = cycles[0]
# qpos46 = mirror_cycle(qpos46)
# qpos46 = extend_and_smooth_46(qpos46, repeat=10)
# qvel46 = compute_velocity(qpos46)


# qpos52, qvel52 = expand_beta(qpos46, qvel46)
# qpos53 = convert_qpos52_to_mj53(qpos52)
# viz.animate_sequence(qpos53, interval=0.02)


input_dir = 'C:/Users/YAKEz/OneDrive/Desktop/MFG_Musculoskelet_V10/ik_data/louis'
output_dir = 'C:/Users/YAKEz/OneDrive/Desktop/MFG_Musculoskelet_V10/louis_trajs'
batch_process_trajs(input_dir, output_dir)

