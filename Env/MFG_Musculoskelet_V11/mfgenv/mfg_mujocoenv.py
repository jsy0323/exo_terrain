import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Dict, Any, Tuple, List, Mapping

import numpy as np
import mujoco
import mujoco.viewer as mujviewer
from gymnasium import spaces

from .mfg_baseenv import BaseEnv
from .mujoco_utils import (
    load_mujoco_model,
    check_invalid_names,
    parse_actuator_prm_from_xml,
    compute_muscle_volumes
)
from .common_utils import convert_ref_traj_qpos
from .human_state import HumanStateMap

from datetime import datetime, timedelta

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow is optional for headless runs
    Image = None

_PLOT_LIB = None  # Lazy matplotlib handle shared across instances
_REF_CURRENT_RGBA = np.array([0.9, 0.15, 0.15, 0.95], dtype=np.float64)
_REF_CURRENT_RGBA.setflags(write=False)
_REF_TRAIL_RGBA = np.array([0.6, 0.6, 0.6, 0.7], dtype=np.float64)
_REF_TRAIL_RGBA.setflags(write=False)
_DEFAULT_REF_SITES = (
    "pelvis_sensor",
    "hand_r_sensor",
    "hand_l_sensor",
    "toes_r_sensor",
    "toes_l_sensor",
    "head_sensor",
)


def _require_plotting():
    """Load matplotlib lazily to avoid import overhead in headless execution."""
    global _PLOT_LIB
    if _PLOT_LIB is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433 - intentionally local import
        _PLOT_LIB = plt
    return _PLOT_LIB

logger = logging.getLogger(__name__)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(ch)

class MuJoCoEnv(BaseEnv):
    """
    A MuJoCo-based Gymnasium environment.

    Attributes
    ----------
    model : mujoco.MjModel
        Compiled MuJoCo model with only the actuators you intend to control.
    data : mujoco.MjData
        Simulation state corresponding to 'model'.
    action_space : spaces.Box
        Continuous controls matching 'model.actuator[i].ctrlrange'.
    observation_space : spaces.Box
        Concatenated [qpos, qvel] with ±inf bounds by default.
    frame_skip : int
        Number of physics steps per call to 'step()'.
    auto_render : bool
        If True, automatically calls 'render()' after 'reset()' and 'step()'.
    num_dofs : int
        Number of generalized coordinates (model.nv).
    num_actuators : int
        Number of actuators (model.nu).
    jnt_names : List[str]
        Names of all joints.
    actuator_names : List[str]
        Names of all actuators.

    Notes
    -----
    * Invariants derived from the MJCF (actuator ranges, default state, parsed
      actuator parameters, etc.) are cached at the class level. Recreating many
      environments – for example in vectorised training across Windows or Linux
      workers – reuses this cache to avoid repeated XML parsing and Python-side
      bookkeeping.
    * Optional tooling (matplotlib history plots, reference-marker buffers) is
      initialised lazily. Headless training that never requests those features
      therefore incurs virtually no extra import or allocation overhead.
    """

    @dataclass
    class _ModelMeta:
        ctrl_low: np.ndarray
        ctrl_high: np.ndarray
        init_qpos: np.ndarray
        init_qvel: np.ndarray
        jnt_names: Tuple[str, ...]
        actuator_names: Tuple[str, ...]
        actuator_prms: Tuple[Tuple[str, Mapping[str, Any]], ...]
        muscle_volumes: Optional[np.ndarray]
        total_mass: float

    _MODEL_META_CACHE: Dict[Tuple[str, bool, bool], _ModelMeta] = {}
    _XML_ACTUATOR_CACHE: Dict[str, Dict[str, Mapping[str, Any]]] = {}

    def __init__(self, config: Dict[str, Any], render_mode: Optional[str] = None, **kwargs):
        """
        Initialize the MuJoCo environment.

        Args
        ----
        config : dict
            Required keys:
              - 'model_path' (str): Path to MJCF XML file.
              - 'default_frame_skip' (int): Physics steps per env.step().
            Optional keys:
              - 'render_mode' (str or None): One of BaseEnv.metadata["render_modes"].
              - 'timestep' (float): Override the XML integrator timestep.
              - 'camera' (int or str): Camera index or name for offscreen.
              - 'use_muscle' (bool): If True, keep group 1+3 actuators; else 1+2.
              - 'auto_render' (bool): If True, call render() automatically.
              - 'render_reference' (dict): Optional settings for reference markers
                    {\"enabled\": bool, \"sites\": [...], \"history_len\": int,
                     \"future_len\": int, \"stride\": int}.

        Raises
        ------
        KeyError
            If required config keys are missing.
        ValueError
            If 'default_frame_skip' < 1.
        RuntimeError
            If model loading or compilation fails.
        """
        logger.debug("Initializing MuJoCoEnv")
        # ---------------------------------------------------------------
        # Validate config keys
        # ---------------------------------------------------------------
        if "model_path" not in config:
            raise KeyError("Config must contain 'model_path'.")
        if "default_frame_skip" not in config:
            raise KeyError("Config must contain 'default_frame_skip'.")
            
        model_path = config["model_path"]
        default_fs = int(config["default_frame_skip"])
        if default_fs < 1:
            raise ValueError(f"default_frame_skip must be >= 1, got {default_fs}.")
        
        rm          = render_mode if render_mode is not None else config.get("render_mode", None)
        timestep    = config.get("timestep", None)
        camera      = config.get("camera", 0)
        use_muscle  = bool(config.get("use_muscle", False))
        load_mesh   = bool(config.get("load_mesh", False))
        auto_render = bool(config.get("auto_render", False)) if rm is not None else False
        
        self.enable_plthist  = bool(config.get("enable_plthist", False))
        self.plthist_len     = int(config.get("plthist_len", 200))
        self.history_stride  = int(config.get("history_stride", 1))
        self.validate_obs    = bool(config.get("validate_obs", False))
        self.return_obs_copy = bool(config.get("return_obs_copy", False))
        self._do_clip_action = bool(config.get("clip_action", False))

        render_options = dict(config.get("render_options", {}) or {})
        render_reference_cfg = dict(config.get("render_reference", {}) or {})
        render_options.setdefault("render_reference", render_reference_cfg)

        # ---------------------------------------------------------------
        # Initialize BaseEnv (sets render_mode, metadata, seed stub)
        # ---------------------------------------------------------------
        super().__init__(render_mode=rm, auto_render=auto_render, render_options=render_options)

        # ---------------------------------------------------------------
        # Load & compile model/data, filtering actuators by use_muscle
        # ---------------------------------------------------------------
        try:
            self.model, self.data = load_mujoco_model(model_path, use_muscle, load_mesh)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model: {e}")

        self.human_state_map = HumanStateMap(self.model)
        resolved_model_path = str(Path(model_path).expanduser().resolve())
        cache_key = (resolved_model_path, use_muscle, load_mesh)

        # ---------------------------------------------------------------
        # Cache XML-derived actuator parameters across instances
        # ---------------------------------------------------------------
        actuator_param_cache = self._XML_ACTUATOR_CACHE.get(resolved_model_path)
        if actuator_param_cache is None:
            raw_params = parse_actuator_prm_from_xml(model_path)
            actuator_param_cache = {
                name: MappingProxyType(dict(payload)) for name, payload in raw_params.items()
            }
            self._XML_ACTUATOR_CACHE[resolved_model_path] = actuator_param_cache

        # ---------------------------------------------------------------
        # Cache model invariants (ctrl range, default state, naming, etc.)
        # ---------------------------------------------------------------
        meta = self._MODEL_META_CACHE.get(cache_key)
        if meta is None:
            jnt_names = tuple(self.model.joint(i).name for i in range(int(self.model.njnt)))
            actuator_names = tuple(self.model.actuator(i).name for i in range(int(self.model.nu)))
            check_invalid_names([n for n in jnt_names if not n.startswith('interface_')],
                                list(actuator_names), use_muscle)

            ctrl_low = np.array(
                [self.model.actuator(i).ctrlrange[0] for i in range(self.model.nu)],
                dtype=np.float32,
            )
            ctrl_high = np.array(
                [self.model.actuator(i).ctrlrange[1] for i in range(self.model.nu)],
                dtype=np.float32,
            )

            init_qpos = self.data.qpos.copy()
            init_qvel = self.data.qvel.copy()

            actuator_prms = tuple(
                (name, actuator_param_cache[name])
                for name in actuator_names
                if name in actuator_param_cache
            )

            muscle_volumes = None
            if use_muscle:
                muscle_volumes = compute_muscle_volumes(
                    {name: dict(prm) for name, prm in actuator_prms},
                    list(actuator_names),
                )

            ctrl_low.setflags(write=False)
            ctrl_high.setflags(write=False)
            init_qpos.setflags(write=False)
            init_qvel.setflags(write=False)
            if muscle_volumes is not None:
                muscle_volumes.setflags(write=False)

            meta = self._ModelMeta(
                ctrl_low=ctrl_low,
                ctrl_high=ctrl_high,
                init_qpos=init_qpos,
                init_qvel=init_qvel,
                jnt_names=jnt_names,
                actuator_names=actuator_names,
                actuator_prms=actuator_prms,
                muscle_volumes=muscle_volumes,
                total_mass=float(self.model.body_mass[1:].sum()),
            )
            self._MODEL_META_CACHE[cache_key] = meta

        # Instance-level copies depending on configuration
        self._init_qpos = meta.init_qpos.copy()
        self._init_qvel = meta.init_qvel.copy()

        self.jnt_names = list(meta.jnt_names)
        self.actuator_names = list(meta.actuator_names)
        self.total_mass = meta.total_mass

        self.actuator_prms = {name: dict(prm) for name, prm in meta.actuator_prms}
        if use_muscle and meta.muscle_volumes is not None:
            self.muscle_volumes = meta.muscle_volumes.copy()
        else:
            self.muscle_volumes = None

        self._build_actuator_type_masks()

        # Override integrator timestep if provided
        if timestep is not None:
            self.model.opt.timestep = float(timestep)
        self.opt_time = float(self.model.opt.timestep)

        # ---------------------------------------------------------------
        # Basic dimensions & action space
        # ---------------------------------------------------------------
        self.num_dofs = int(self.model.nv)
        self.num_joints = len(self.jnt_names)
        self.num_actuators = len(self.actuator_names)

        self.action_space = spaces.Box(low=meta.ctrl_low, high=meta.ctrl_high, dtype=np.float32)
        self._ctrl_low = meta.ctrl_low
        self._ctrl_high = meta.ctrl_high

        # ---------------------------------------------------------------
        # Reference trajectory visualisation configuration
        # ---------------------------------------------------------------
        ref_vis_cfg = render_reference_cfg
        self._show_reference_markers = bool(ref_vis_cfg.get("enabled", False))
        if self._show_reference_markers or (rm == "human"):
            self._ref_site_names = tuple(ref_vis_cfg.get("sites", _DEFAULT_REF_SITES))
            self._ref_history_len = max(0, int(ref_vis_cfg.get("history_len", 20)))
            self._ref_future_len = max(0, int(ref_vis_cfg.get("future_len", 20)))
            self._ref_stride = max(1, int(ref_vis_cfg.get("stride", 5)))
        else:
            self._ref_site_names = ()
            self._ref_history_len = 0
            self._ref_future_len = 0
            self._ref_stride = 1
        self._ref_current_radius = 0.05
        self._ref_trail_radius = 0.025
        self._ref_current_rgba = _REF_CURRENT_RGBA
        self._ref_trail_rgba = _REF_TRAIL_RGBA
        self._ref_site_ids: Optional[List[int]] = None
        self._ghost_data: Optional[mujoco.MjData] = None

        # ---------------------------------------------------------------
        # Interaction flags & frame skip
        # ---------------------------------------------------------------
        self.frame_skip     = default_fs
        self.paused         = False
        self.follow         = False
        self.disable_reset  = False
        self.camera         = camera
        
        # ---------------------------------------------------------------
        # Build action_space from ctrlrange of each actuator
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        # Build observation_space (qpos + qvel)
        # ---------------------------------------------------------------
        inf = np.inf
        pos_low  = -inf * np.ones_like(self._init_qpos, dtype=np.float32)
        pos_high = +inf * np.ones_like(self._init_qpos, dtype=np.float32)
        vel_low  = -inf * np.ones_like(self._init_qvel, dtype=np.float32)
        vel_high = +inf * np.ones_like(self._init_qvel, dtype=np.float32)
        obs_low  = np.concatenate([pos_low, vel_low], axis=0)
        obs_high = np.concatenate([pos_high, vel_high], axis=0)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        self._obs_dim = self._init_qpos.size + self._init_qvel.size
        self._obs_buf = np.empty(self._obs_dim, dtype=np.float32)

        # ---------------------------------------------------------------
        # Prepare history buffers for qpos, qvel, ctrl
        # ---------------------------------------------------------------
        if self.enable_plthist and self.plthist_len > 0:
            from collections import deque
            self._hist_qpos = deque(maxlen=self.plthist_len)
            self._hist_qvel = deque(maxlen=self.plthist_len)
            self._hist_ctrl = deque(maxlen=self.plthist_len)
            self._hist_step = 0
        else:
            self._hist_qpos = self._hist_qvel = self._hist_ctrl = None
            self._hist_step = 0
    
        logger.debug(
            "MuJoCoEnv initialized: nv=%d, nu=%d, frame_skip=%d, timestep=%.6f",
            self.num_dofs, self.num_actuators, self.frame_skip, self.opt_time
        )
        
    def _build_actuator_type_masks(self):
        names = list(self.actuator_names)
        nu = len(names)
    
        prms = getattr(self, "actuator_prms", {})
        if not isinstance(prms, dict):
            prms = {}
    
        mus_mask = np.zeros(nu, dtype=bool)
        pos_mask = np.zeros(nu, dtype=bool)
    
        for i, nm in enumerate(names):
            p = prms.get(nm)
            t = (p or {}).get("type", None)
            if t == "muscle":
                mus_mask[i] = True
            elif t == "position":
                pos_mask[i] = True
    
        self.muscle_mask    = mus_mask
        self.muscle_indices = np.nonzero(mus_mask)[0].astype(np.int64, copy=False)
        self.num_muscles    = int(mus_mask.sum())
        
        self.position_mask    = pos_mask
        self.position_indices = np.nonzero(pos_mask)[0].astype(np.int64, copy=False)

    def create_viewer(self) -> None:
        """
        Instantiate the MuJoCo viewer or offscreen renderer based on 'self.render_mode'.
        Called by BaseEnv.render() on first invocation.

        - "human": launches a passive GLFW viewer and registers 'key_callback'.
        - "rgb_array": creates an offscreen Renderer and primes the first frame.
        - None: ensures no viewer or renderer is active.

        Raises
        ------
        RuntimeError
            If 'render_mode' is not one of {None, "human", "rgb_array"} or
            if creation of the viewer/renderer fails.
        """
        mode = self.render_mode
        if mode not in (None, "human", "rgb_array"):
            raise RuntimeError(f"Unsupported render_mode: {mode}")
            
        if mode == "human" and getattr(self, "viewer", None) is not None:
            return
        if mode == "rgb_array" and getattr(self, "renderer", None) is not None:
            return
        
        if getattr(self, "viewer", None) is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        if getattr(self, "renderer", None) is not None:
            self.renderer = None
            
        if mode == "human":
            try:
                self.viewer = mujviewer.launch_passive(
                    self.model,
                    self.data,
                    key_callback=self.key_callback
                )
                cam = self.viewer.cam
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = min(1, self.model.nbody - 1)
                cam.fixedcamid = -1
                cam.distance  = 5.0
                cam.azimuth   = 135.0
                cam.elevation = -20.0
            except Exception as e:
                raise RuntimeError(f"Failed to launch human-mode viewer: {e}")
                
        elif mode == "rgb_array":
            try:
                self.renderer = mujoco.Renderer(self.model)
                mujoco.mj_forward(self.model, self.data)
                self.renderer.update_scene(self.data)
            except Exception as e:
                raise RuntimeError(f"Failed to create offscreen renderer: {e}")
                
        else:
            self.viewer = None
            self.renderer = None

    def key_callback(self, keycode: int) -> None:
        """
        Handle keyboard events in the human-mode viewer.

        Supported keys:
          - Space (" "): Toggle pause/resume of stepping.
          - "R":         Reset the environment (if 'disable_reset=False').
          - "F":         Toggle camera follow mode.
          - "M":         Toggle 'disable_reset' flag.
          - "D":         Print the first three qpos/qvel for debugging.
          - "S":         Save an RGB screenshot (only in "rgb_array" mode).
          - "G":         Trigger history plotting (qpos, qvel, ctrl).
          - "C":         Toggle reference markers (current + history/future trajectories).

        Notes
        -----
        This callback is only active when 'render_mode=="human"'.
        """
        try:
            c = chr(keycode)
        except Exception:
            return
        
        cu = c.upper()
        if c == " ":
            self.paused = not self.paused
            print(f"[MuJoCoEnv] Paused = {self.paused}")
            return

        elif cu == "R":
            if not self.disable_reset:
                print("[MuJoCoEnv] Resetting environment.")
                self.reset()
            else:
                print("[MuJoCoEnv] Reset is disabled.")
            return

        elif cu == "F":
            self.follow = not self.follow
            print(f"[MuJoCoEnv] Follow = {self.follow}")
            return

        elif cu == "M":
            self.disable_reset = not self.disable_reset
            print(f"[MuJoCoEnv] Disable reset = {self.disable_reset}")
            return

        elif cu == "D":
            qpos_sample = self.data.qpos[:3] if self.num_dofs >= 3 else self.data.qpos
            qvel_sample = self.data.qvel[:3] if self.num_dofs >= 3 else self.data.qvel
            print(f"[MuJoCoEnv] qpos[:3]={np.array2string(qpos_sample, precision=4)}, "
                  f"qvel[:3]={np.array2string(qvel_sample, precision=4)}")
            return

        elif cu == "S" and self.render_mode == "rgb_array":
            if Image is None:
                print("[MuJoCoEnv] Pillow is not installed; cannot save screenshots.")
                return
            if getattr(self, "renderer", None) is None:
                self.create_viewer()
            if self.renderer is not None:
                img = self.renderer.render()
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"screenshot_{ts}.png"
                Image.fromarray(img).save(path)
                print(f"[MuJoCoEnv] Screenshot saved: {path}")
            return

        elif cu == "C":
            self._show_reference_markers = not self._show_reference_markers
            state = "ON" if self._show_reference_markers else "OFF"
            print(f"[MuJoCoEnv] Reference markers {state}")
            if not self._show_reference_markers and getattr(self, "viewer", None) is not None:
                self.viewer.user_scn.ngeom = 0
            return

        elif cu == "G":
            self._plot_history()
            return
    
    def _plot_history(self):
        """
        Plot and save joint position, velocity, and actuator control histories.

        This method creates a 3-panel figure showing:
          1. The first two joint positions over time.
          2. The first two joint velocities over time.
          3. The first two actuator commands over time.

        The plot is saved as a timestamped PNG in the 'history_plots/' folder,
        which is created if it does not already exist.

        Raises
        ------
        RuntimeError
            If no history data is available when attempting to plot.
        """
        if not (self.enable_plthist and self.plthist_len > 0 and self._hist_qpos is not None):
            print("[MuJoCoEnv] History plotting is disabled.")
            return

        if not self._hist_qpos or not self._hist_qvel or not self._hist_ctrl:
            print("[MuJoCoEnv] No history to plot.")
            return

        plt = _require_plotting()

        # Convert history deques to contiguous arrays once to avoid repeated small copies
        qpos_arr = np.asarray(self._hist_qpos, dtype=np.float32)
        qvel_arr = np.asarray(self._hist_qvel, dtype=np.float32)
        ctrl_arr = np.asarray(self._hist_ctrl, dtype=np.float32)
        
        n_steps = qpos_arr.shape[0]
        if n_steps < 2:
            print("[MuJoCoEnv] Not enough history to plot.")
            return
        MAX_POINTS = 2000
        if n_steps > MAX_POINTS:
            stride = max(1, n_steps // MAX_POINTS)
            qpos_arr = qpos_arr[::stride]
            qvel_arr = qvel_arr[::stride]
            ctrl_arr = ctrl_arr[::stride]
            n_steps  = qpos_arr.shape[0]
        steps = np.arange(n_steps, dtype=int)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Simulation History", fontsize=16)

        # Plot first two qpos dimensions
        n_q = min(2, qpos_arr.shape[1] if qpos_arr.ndim > 1 else 1)
        for i in range(n_q):
            axes[0].plot(steps, qpos_arr[:, i], label=f"qpos[{i}]")
        axes[0].set_ylabel("qpos")
        axes[0].legend(loc="upper right", fontsize=9)
        axes[0].grid(True)
    
        # Plot first two qvel dimensions
        n_v = min(2, qvel_arr.shape[1] if qvel_arr.ndim > 1 else 1)
        for i in range(n_v):
            axes[1].plot(steps, qvel_arr[:, i], label=f"qvel[{i}]")
        axes[1].set_ylabel("qvel")
        axes[1].legend(loc="upper right", fontsize=9)
        axes[1].grid(True)
    
        # Plot control history: prefer legacy indices but fall back automatically when absent
        if ctrl_arr.ndim == 1:
            ctrl_arr = ctrl_arr[:, np.newaxis]
        num_ctrl = ctrl_arr.shape[1]
        preferred = (12, 13, 33, 35)
        indices = [idx for idx in preferred if idx < num_ctrl]
        if not indices and num_ctrl > 0:
            indices = list(range(min(num_ctrl, 2)))

        if indices:
            for j in indices:
                axes[2].plot(steps, ctrl_arr[:, j], label=f"ctrl[{j}]")
            axes[2].set_ylabel("ctrl")
            axes[2].set_xlabel("Step")
            axes[2].legend(loc="upper right", fontsize=9)
            axes[2].grid(True)
        else:
            axes[2].text(0.5, 0.5, "No ctrl history", ha="center", va="center", transform=axes[2].transAxes)
            axes[2].set_axis_off()
    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
        folder = "history_plots"
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(folder, f"history_{timestamp}.png")
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"[MuJoCoEnv] History plot saved to '{save_path}'.")

    def _get_simulation_state(self):
        """
        Provide the current MjData to the offscreen renderer.
        Called by BaseEnv.render() in "rgb_array" mode.
        """
        return self.data

    def _get_current_camera(self):
        """
        Provide the camera index or name for offscreen rendering.
        Called by BaseEnv.render() in "rgb_array" mode.
        """
        cam = int(self.camera)
        if cam < -1 or cam >= int(self.model.ncam):
            return -1
        return cam

    # ------------------------------------------------------------------
    # Rendering (human mode reference markers)
    # ------------------------------------------------------------------
    def _render_human(self):
        if self.viewer is None:
            raise RuntimeError("Human mode requested but viewer is not initialised.")

        scene = self.viewer.user_scn
        if self._show_reference_markers and self._ensure_reference_sites() and getattr(self, "ref_traj", None) is not None:
            self._populate_reference_scene(scene)
        else:
            scene.ngeom = 0

        return super()._render_human()

    # ------------------------------------------------------------------
    # Reference trajectory visualisation helpers
    # ------------------------------------------------------------------
    def _ensure_reference_sites(self) -> bool:
        """Resolve configured site names to numeric IDs (once)."""
        if self._ref_site_ids is not None:
            return True

        ids: List[int] = []
        for name in self._ref_site_names:
            try:
                sid = self.model.site(name).id
            except Exception:
                self.logger.debug("Reference site '%s' not found; skipping.", name)
                continue
            ids.append(sid)

        if not ids:
            self.logger.debug("No valid reference sites found; disabling reference overlay.")
            self._ref_site_names = ()
            return False

        self._ref_site_ids = ids
        return True

    def _get_ghost_data(self) -> mujoco.MjData:
        """Lazy allocation of scratch mjData used for reference forward kinematics."""
        ghost = self._ghost_data
        if ghost is not None:
            return ghost
        ghost = mujoco.MjData(self.model)
        self._ghost_data = ghost
        self.register_resource(lambda: setattr(self, "_ghost_data", None))
        return ghost

    def _compute_site_positions(self, qpos_vec: np.ndarray) -> np.ndarray:
        """Forward reference qpos and return site positions for configured markers."""
        ghost = self._get_ghost_data()
        qpos_full, _ = self.human_state_map.expand(convert_ref_traj_qpos(qpos_vec))

        # The mapping supplies both human coordinates and neutral cuff frames.
        np.copyto(ghost.qpos, qpos_full)
        ghost.qvel[:] = 0.0
        mujoco.mj_fwdPosition(self.model, ghost)

        idx = self._ref_site_ids or []
        if not idx:
            return np.empty((0, 3), dtype=np.float64)
        return ghost.site_xpos[idx, :].copy()

    def _populate_reference_scene(self, scene: mujoco.MjvScene) -> None:
        """Populate ``scene`` with current/history/future reference markers."""
        ref = getattr(self, "ref_traj", None)
        if ref is None or not self._ensure_reference_sites():
            scene.ngeom = 0
            return

        scene.ngeom = 0

        mat = np.eye(3, dtype=np.float64).ravel()

        # Draw history markers (older frames first)
        if self._ref_history_len > 0:
            available = ref._pos
            if available > 0:
                max_steps = min(self._ref_history_len, available)
                for offset in range(self._ref_stride, max_steps + 1, self._ref_stride):
                    qpos_hist, _ = ref.get_reference_trajectories(-offset)
                    points = self._compute_site_positions(qpos_hist)
                    self._add_points_to_scene(scene, points, self._ref_trail_radius, self._ref_trail_rgba, mat)

        # Draw future markers
        if self._ref_future_len > 0:
            remaining = ref.traj_frames - ref._pos - 1
            if remaining > 0:
                max_steps = min(self._ref_future_len, remaining)
                for offset in range(self._ref_stride, max_steps + 1, self._ref_stride):
                    qpos_future, _ = ref.get_reference_trajectories(offset)
                    points = self._compute_site_positions(qpos_future)
                    self._add_points_to_scene(scene, points, self._ref_trail_radius, self._ref_trail_rgba, mat)

        # Draw current markers last so they stand out.
        qpos_curr = ref.get_qpos()
        current_points = self._compute_site_positions(qpos_curr)
        self._add_points_to_scene(scene, current_points, self._ref_current_radius, self._ref_current_rgba, mat)

    def _add_points_to_scene(
        self,
        scene: mujoco.MjvScene,
        points: np.ndarray,
        radius: float,
        rgba: np.ndarray,
        mat: np.ndarray,
    ) -> None:
        """Append spherical markers for each point into ``scene``."""
        if points.size == 0:
            return
        size = np.array([radius, radius, radius], dtype=np.float64)
        for pos in points:
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                pos.astype(np.float64, copy=False),
                mat,
                rgba,
            )
            scene.ngeom += 1
    
    def _clear_history(self) -> None:
        """
        Clear all stored simulation history buffers.
        """
        if getattr(self, "_hist_qpos", None) is not None:
            self._hist_qpos.clear()
            self._hist_qvel.clear()
            self._hist_ctrl.clear()
            self._hist_step = 0
        logger.debug("Simulation history buffers have been cleared.")
    
    def _cleanup_plots(
            self,
            folder: str = "history_plots",
            max_age_days: int = 3
            ) -> None:
        """
        Remove old plot files to prevent disk bloat.

        Scans the specified `folder` for files and deletes any whose
        modification time is older than `max_age_days`. If the folder
        does not exist, the method exits silently.

        Args:
            folder (str): Directory where historical plots are saved.
                Defaults to "history_plots".
            max_age_days (int): Files older than this many days will
                be deleted. Defaults to 7 days.

        Returns:
            None
        """
        try:
            if not os.path.isdir(folder):
                logger.debug("No plot directory found at '%s'; skipping cleanup.", folder)
                return

            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            cutoff_ts = cutoff_time.timestamp()
            removed_files = 0

            with os.scandir(folder) as it:
                for entry in it:
                    try:
                        if not entry.is_file():
                            continue
                        if entry.stat().st_mtime < cutoff_ts:
                            os.remove(entry.path)
                            removed_files += 1
                            logger.debug("Removed old plot file: %s", entry.path)
                    except Exception as err:
                        logger.warning("Could not remove file '%s': %s", entry.path, err)
    
            logger.debug(
                "Cleanup complete: %d file(s) older than %d day(s) removed from '%s'.",
                removed_files, max_age_days, folder
            )
        except Exception as err:
            logger.warning("Plot cleanup failed for '%s': %s", folder, err)
    
    def reset(
            self, 
            *, 
            seed: Optional[int] = None, 
            options: Dict[str, Any] = None
            ) -> Tuple[np.ndarray, dict]:
        """
        Reset the MuJoCo simulation to its initial state (or a custom state).

        This implements the Gymnasium API:
          - Uses 'seed' to reseed RNGs.
          - Accepts an 'options' dict; if it contains "custom_init_state",
            that dict may supply "qpos" and/or "qvel" arrays to override
            the factory defaults.
          - Clears internal history buffers.
          - Primes the offscreen renderer if in "rgb_array" mode.
          - Auto‐renders the first frame if enabled.

        Args:
            seed: Optional integer seed to make resets reproducible.
            options: Optional dict. May contain:
                "custom_init_state": {
                    "qpos": np.ndarray of length nv,
                    "qvel": np.ndarray of length nv
                }

        Returns:
            obs (np.ndarray): The initial observation, concatenated [qpos, qvel].
            info (dict): Empty dict (override in subclasses if needed).

        Raises:
            ValueError: If a provided custom_init_state vector has incorrect length.
        """
        logger.debug("Resetting environment (seed=%s)", seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Parse options and seed
        if seed is not None:
            self._seed(seed)
        opts = options or {}
        custom = opts.get("custom_init_state", None)

        # Restore qpos/qvel
        if custom is not None:
            if "qpos" in custom:
                qpos = np.asarray(custom["qpos"], dtype=np.float64)
                if qpos.shape != self._init_qpos.shape:
                    raise ValueError(f"custom qpos shape {qpos.shape} != {self._init_qpos.shape}")
                np.copyto(self.data.qpos, qpos)
            else:
                np.copyto(self.data.qpos, self._init_qpos)

            if "qvel" in custom:
                qvel = np.asarray(custom["qvel"], dtype=np.float64)
                if qvel.shape != self._init_qvel.shape:
                    raise ValueError(f"custom qvel shape {qvel.shape} != {self._init_qvel.shape}")
                np.copyto(self.data.qvel, qvel)
            else:
                np.copyto(self.data.qvel, self._init_qvel)
        else:
            np.copyto(self.data.qpos, self._init_qpos)
            np.copyto(self.data.qvel, self._init_qvel)
            
        self.data.ctrl.fill(0.0)

        # Forward pass to update geometry, sensors, etc.
        mujoco.mj_forward(self.model, self.data)
        
        # Clear any stored history
        self._clear_history()

        # Update offscreen renderer if needed
        if self.render_mode == "rgb_array" and getattr(self, "renderer", None) is not None:
            self.renderer.update_scene(self.data)

        # Visualize this reset state
        if self.render_mode is not None and self.auto_render:
            _ = self.render()

        # Return dummy placeholders; subclasses must override to return real obs and info
        obs, components = self._get_obs()
        if self.validate_obs and (not self.observation_space.contains(obs)):
            raise ValueError(f"Observation {obs} out of bounds for {self.observation_space}")
        
        info = {"obs_components": components}
        logger.debug("Reset done, obs shape=%s", obs.shape)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Advance the simulation by one environment step using the given action.

        This method implements the Gymnasium step API:
          1. Validate the action’s shape and bounds against 'self.action_space'.
          2. Apply the action to 'self.data.ctrl' and perform 'self.frame_skip' physics steps.
          3. Record qpos, qvel, and ctrl into history buffers.
          4. Update camera (human mode) and renderer (rgb_array mode) as needed.
          5. Auto‐render the frame if 'self.auto_render' is True.
          6. Construct the next observation, then return
             (obs, reward, terminated, truncated, info).

        Args
        ----
        action : np.ndarray of shape (num_actuators,)
            Continuous control inputs for each actuator.

        Returns
        -------
        obs        : np.ndarray
            Concatenated [qpos, qvel] after the step.
        reward     : float
            Placeholder reward (always 0.0 here; override in subclasses).
        terminated : bool
            Episode termination flag (always False here; override in subclasses).
        truncated  : bool
            Episode truncation flag (always False here; override in subclasses).
        info       : dict
            Auxiliary information (empty by default; override in subclasses).

        Raises
        ------
        ValueError
            If 'action' has incorrect shape or contains out‐of‐bounds values.
        """
        logger.debug("Starting step")
        
        pause_sleep = 0.01
        if self.paused:
            fps = float(self.metadata.get("render_fps", 50) or 50)
            pause_sleep = max(1.0 / fps, pause_sleep)
        while self.paused:
            if self.render_mode is not None and self.auto_render:
                _ = self.render(force=True)
            time.sleep(pause_sleep)

        act = action
        if not (isinstance(act, np.ndarray) and act.dtype == np.float32 
                and act.flags.c_contiguous and act.shape == (self.num_actuators,)):
            act = np.asarray(act, dtype=np.float32, order="C")
        
        if self._do_clip_action:
           if (act < self._ctrl_low).any() or (act > self._ctrl_high).any():
               np.clip(act, self._ctrl_low, self._ctrl_high, out=act)
            
        # Physics stepping
        if not self.paused:
            d = self.data
            m = self.model
            fs = self.frame_skip
            d.ctrl[:] = act
            for _ in range(fs):
                mujoco.mj_step(m,d)
            
        if self._hist_qpos is not None:
            self._hist_step += 1
            if (self._hist_step % self.history_stride) == 0:
                self._hist_qpos.append(self.data.qpos.astype(np.float32, copy=True))
                self._hist_qvel.append(self.data.qvel.astype(np.float32, copy=True))
                self._hist_ctrl.append(act.copy()) 

        # Human‐mode camera follow
        if self.render_mode == "human" and self.follow and getattr(self, "viewer", None) is not None:
            cam = self.viewer.cam
            q0 = self.data.qpos
            cam.lookat[0] = float(q0[0])
            cam.lookat[1] = float(q0[1])
            cam.lookat[2] = float(q0[2])

        # Offscreen rendering update
        if self.render_mode == "rgb_array" and getattr(self, "renderer", None) is not None:
            self.renderer.update_scene(self.data)

        # Auto‐render if enabled
        if self.render_mode is not None and self.auto_render:
            _ = self.render()

        # Build observation
        obs, components = self._get_obs()
        if self.validate_obs and (not self.observation_space.contains(obs)):
            raise ValueError(f"Observation {obs} out of bounds for {self.observation_space}")
        
        # Return placeholders for reward/termination/info
        reward     = 0.0
        terminated = False
        truncated  = False
        info       = {"obs_components": components}
        
        logger.debug("Step completed: obs_shape=%s", getattr(obs, "shape", None))
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """
        Unify functions that construct observation vectors for easy subclass coverage.
        """
        if not hasattr(self, "_obs_buf"):
            nq = self.data.qpos.size
            nv = self.data.qvel.size
            self._obs_dim = nq + nv
            self._obs_buf = np.empty(self._obs_dim, dtype=np.float32)
            
        nq = self._obs_dim - self.data.qvel.size
        buf = self._obs_buf
        
        buf[:nq] = self.data.qpos
        buf[nq:] = self.data.qvel
    
        return (buf.copy() if getattr(self, "return_obs_copy", False) else buf), {}
    
    def _ensure_joint_site_metadata(self):
        """
        Populate self._joint_sites and self._pelvis_gyro exactly once,
        mirroring the logic in get_joint_kinematics().
        """
        if hasattr(self, "_joint_sites") and hasattr(self, "_pelvis_gyro"):
            return

        model = self.model

        try:
            pelvis_sid = model.site("pelvis_sensor").id
        except Exception:
            pelvis_sid = -1
    
        # temp[sid] = {"name": str, "vel": (adr, dim) or None, "gyr": (adr, dim) or None}
        temp = {}
    
        for si in range(model.nsensor):
            if model.sensor_objtype[si] != mujoco.mjtObj.mjOBJ_SITE:
                continue
    
            sid   = int(model.sensor_objid[si])
            stype = int(model.sensor_type[si])
            adr   = int(model.sensor_adr[si])
            dim   = int(model.sensor_dim[si])
    
            if sid not in temp:
                sname = model.site(sid).name
                if isinstance(sname, bytes):
                    sname = sname.decode()
                temp[sid] = {"name": sname, "vel": None, "gyr": None}
    
            if stype == mujoco.mjtSensor.mjSENS_VELOCIMETER:
                temp[sid]["vel"] = (adr, dim)
            elif stype == mujoco.mjtSensor.mjSENS_GYRO:
                temp[sid]["gyr"] = (adr, dim)
    
        pelvis_gyro = (None, 0)
        if pelvis_sid in temp and temp[pelvis_sid]["gyr"] is not None:
            pelvis_gyro = temp[pelvis_sid]["gyr"]
    
        joint_sites = []
        for sid, rec in temp.items():
            if sid == pelvis_sid:
                continue
            if rec["vel"] is not None:
                gyr = rec["gyr"] if rec["gyr"] is not None else (None, 0)
                joint_sites.append((sid, rec["name"], rec["vel"], gyr))
    
        if not joint_sites:
            raise ValueError("No joint-site sensors found in the model (excluding pelvis_sensor).")
    
        self._joint_sites = joint_sites
        self._pelvis_gyro = pelvis_gyro

    def close(self):
        """
        Close MuJoCo viewer/renderer and release resources.
        """
        logger.debug("Closing MuJoCoEnv and cleaning up resources")
        
        viewer = getattr(self, "viewer", None)
        if viewer is not None:
            try:
                viewer.close()
            except Exception as e:
                logger.debug("Viewer close warning: %s", e)
            self.viewer = None
    
        renderer = getattr(self, "renderer", None)
        if renderer is not None:
            self.renderer = None
        
        self._clear_history()
        try:
            self._cleanup_plots()
        except Exception as e:
            logger.debug("Cleanup plots warning: %s", e)
        
        try:
            super().close()
        except Exception as e:
            logger.debug("BaseEnv close warning: %s", e)
            super().close()
            
        logger.debug("MuJoCoEnv closed successfully")
