# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 20:03:59 2025

@author: YAKE
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import logging

logger = logging.getLogger(__name__)

class TrajectoryManager:
    """
    Manages preprocessed 52‐DOF reference trajectories loaded from .npz files.

    Each .npz must contain:
      - 'qpos': ndarray of shape (52, T)
      - 'qvel': ndarray of shape (52, T)

    On initialization, loads all files from a directory, computes:
      - average forward speed (m/s)
      - foot‐start indicator (0=right, 1=left)
    Filters by speed_range, caches results to avoid repeated I/O,
    and provides accessors to retrieve trajectories and metadata, including:
      - boolean periodicity flags
      - repeat counts and cycle lengths
      - normalized autocorrelation confidence scores

    Attributes:
        qpos_list (Tuple[np.ndarray, ...]): Loaded qpos arrays (read-only).
        qvel_list (Tuple[np.ndarray, ...]): Loaded qvel arrays (read-only).
        speeds_list (np.ndarray): Average speeds for each trajectory (float32).
        foot_starts (np.ndarray): Foot‐start flags for each trajectory (0/1, int8).
        periodic_flags (np.ndarray): True if periodic, else False.
        repeat_counts (np.ndarray): Estimated number of repeats per clip (int32).
        cycle_lengths (np.ndarray): Estimated frames per cycle (>=1, int32).
        period_confidence (np.ndarray): Confidence in periodicity estimate (float32).
        jnt_name (Dict[str,int]): Joint name→index mapping.
        sample_frequency (float): Hz for frame→time conversion.
        speed_range (Tuple[float,float]): Min/max speed filter in m/s.
    """

    # Final 52‐DOF joint names with beta‐DOFs after each mtp joint
    EXPANDED_JOINT_NAMES: List[str] = [
        'pelvis_tz','pelvis_ty','pelvis_tx',
        'pelvis_tilt','pelvis_list','pelvis_rotation',
        'hip_flexion_r','hip_adduction_r','hip_rotation_r',
        'knee_angle_r','ankle_angle_r','subtalar_angle_r','mtp_angle_r',
        'knee_angle_r_beta_translation2',
        'knee_angle_r_beta_translation1',
        'knee_angle_r_beta_rotation1',
        'hip_flexion_l','hip_adduction_l','hip_rotation_l',
        'knee_angle_l','ankle_angle_l','subtalar_angle_l','mtp_angle_l',
        'knee_angle_l_beta_translation2',
        'knee_angle_l_beta_translation1',
        'knee_angle_l_beta_rotation1',
        'lumbar_extension','lumbar_bending','lumbar_rotation',
        'thorax_extension','thorax_bending','thorax_rotation',
        'arm_flex_r','arm_add_r','arm_rot_r',
        'elbow_flex_r','pro_sup_r','wrist_flex_r','wrist_dev_r',
        'arm_flex_l','arm_add_l','arm_rot_l',
        'elbow_flex_l','pro_sup_l','wrist_flex_l','wrist_dev_l',
        'neck_flex','neck_add','neck_rot','head_flex','head_add','head_rot'
    ]
    
    # Prebuilt name→index map
    EXPANDED_JOINT_MAP: Dict[str,int] = {
        name: idx for idx, name in enumerate(EXPANDED_JOINT_NAMES)
    }

    # Class‐level cache to avoid reloading the same directory
    _cache: Dict[str, Dict] = {}

    __slots__ = (
        'path',
        'speed_range',
        'sample_frequency',
        'verbose',
        'jnt_name',
        'qpos_list',
        'qvel_list',
        'speeds_list',
        'foot_starts',
        'periodic_flags',
        'repeat_counts',
        'cycle_lengths',
        'period_confidence',
    )

    # Heuristics for distinguishing periodic vs non-periodic clips
    _AC_PEAK_THRESHOLD: float = 0.7          # normalized autocorr peak
    _HEADING_SPAN_THRESHOLD: float = 0.8     # radians; planar heading range
    _MIN_REPEAT_FRAMES: int = 20             # guard against unrealistically short cycles
    _STEP_ACTIVITY_EPS: float = 1e-4         # meters; ignore tiny jitters

    def __init__(
        self,
        data_path: Union[str, Path],
        speed_range: Tuple[float, float] = (0.0, 3.0),
        sample_frequency: float = 100.0,
        verbose: bool = False
    ):
        """
        Initialize TrajectoryManager.

        Args:
            data_path: Directory containing processed .npz trajectory files.
            speed_range: (min, max) forward speeds in m/s to include.
            sample_frequency: Sampling rate in Hz for frame→time conversion.
            verbose: If True, log detailed loading info.

        Raises:
            ValueError: If data_path is invalid or parameters out of range.
            RuntimeError: If no trajectories pass the filter.
        """
        self.path = Path(data_path)
        if not self.path.is_dir():
            raise ValueError(f"Processed trajectory folder not found: {self.path}")

        if len(speed_range) != 2 or speed_range[0] > speed_range[1]:
            raise ValueError("speed_range must be (min, max) with min <= max")
        if sample_frequency <= 0:
            raise ValueError("sample_frequency must be positive")

        self.speed_range = (float(speed_range[0]), float(speed_range[1]))
        self.sample_frequency = float(sample_frequency)
        self.verbose = bool(verbose)

        # fixed joint mapping
        self.jnt_name = TrajectoryManager.EXPANDED_JOINT_MAP
        self.qpos_list = ()
        self.qvel_list = ()
        self.speeds_list = np.empty(0, dtype=np.float32)
        self.foot_starts = np.empty(0, dtype=np.int8)
        self.periodic_flags = np.empty(0, dtype=np.bool_)
        self.repeat_counts = np.empty(0, dtype=np.int32)
        self.cycle_lengths = np.empty(0, dtype=np.int32)
        self.period_confidence = np.empty(0, dtype=np.float32)

        # load or retrieve from cache
        key = (
            str(self.path.resolve()),
            round(self.speed_range[0], 6),
            round(self.speed_range[1], 6),
            float(self.sample_frequency),
            float(self._AC_PEAK_THRESHOLD),
            float(self._HEADING_SPAN_THRESHOLD),
            int(self._MIN_REPEAT_FRAMES),
            float(self._STEP_ACTIVITY_EPS),
        )
        if key in TrajectoryManager._cache:
            cached = TrajectoryManager._cache[key]
            self.qpos_list         = cached['qpos_list']
            self.qvel_list         = cached['qvel_list']
            self.speeds_list       = cached['speeds_list']
            self.foot_starts       = cached['foot_starts']
            self.periodic_flags    = cached['periodic_flags']
            self.repeat_counts     = cached['repeat_counts']
            self.cycle_lengths     = cached['cycle_lengths']
            self.period_confidence = cached['period_confidence']
        else:
            self._load_and_index()
            TrajectoryManager._cache[key] = {
                'qpos_list':         self.qpos_list,
                'qvel_list':         self.qvel_list,
                'speeds_list':       self.speeds_list,
                'foot_starts':       self.foot_starts,
                'periodic_flags':    self.periodic_flags,
                'repeat_counts':     self.repeat_counts,
                'cycle_lengths':     self.cycle_lengths,
                'period_confidence': self.period_confidence,
            }

    def _load_and_index(self) -> None:
        """
        Private: Load all .npz files, compute average speed and foot-start,
        apply speed_range filter, and populate internal lists.
        """
        files = sorted(self.path.glob("*.npz"))
        if not files:
            raise RuntimeError(f"No .npz files found in {self.path}")

        # Indices for computing speed and foot-start
        lhip   = self.jnt_name['hip_flexion_l']
        rhip   = self.jnt_name['hip_flexion_r']

        qpos_list, qvel_list = [], []
        speeds, foots = [], []
        periodic_flags, repeat_counts = [], []
        cycle_lengths, confidences = [], []

        for f in files:
            try:
                with np.load(f, allow_pickle=False) as data:
                    qpos = data['qpos']
                    qvel = data['qvel']
            except Exception as e:
                logger.warning(f"Skipping {f.name}: cannot load arrays ({e})")
                continue

            if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape != qvel.shape:
                logger.warning(f"Skipping {f.name}: invalid shapes qpos{qpos.shape} qvel{qvel.shape}")
                continue

            T = qpos.shape[1]
            speed = self._estimate_mean_speed(qpos)

            # determine foot-start: 1 if left hip > right hip at start
            foot = int(qpos[lhip, 0] > qpos[rhip, 0])

            analysis = self._analyze_periodicity(qpos)

            # filter by speed
            if self.speed_range[0] <= speed <= self.speed_range[1]:
                qpos_list.append(qpos)
                qvel_list.append(qvel)
                speeds.append(speed)
                foots.append(foot)
                periodic_flags.append(analysis['periodic'])
                repeat_counts.append(analysis['repeat'])
                cycle_lengths.append(analysis['cycle_frames'])
                confidences.append(analysis['confidence'])
                if self.verbose:
                    logger.info(
                        f"Loaded {f.name}: speed={speed:.2f}, foot={foot}, "
                        f"periodic={analysis['periodic']} repeat={analysis['repeat']} "
                        f"cycle={analysis['cycle_frames']} conf={analysis['confidence']:.3f}"
                    )
            else:
                if self.verbose:
                    logger.debug(f"Filtered out {f.name}: speed={speed:.2f} outside {self.speed_range}")

        if not qpos_list:
            raise RuntimeError("No trajectories passed the speed filter.")

        # make arrays read-only
        for arr in qpos_list + qvel_list:
            arr.setflags(write=False)

        self.qpos_list         = tuple(qpos_list)
        self.qvel_list         = tuple(qvel_list)
        self.speeds_list       = np.asarray(speeds, dtype=np.float32)
        self.foot_starts       = np.asarray(foots, dtype=np.int8)
        self.periodic_flags    = np.asarray(periodic_flags, dtype=np.bool_)
        self.repeat_counts     = np.asarray(repeat_counts, dtype=np.int32)
        self.cycle_lengths     = np.asarray(cycle_lengths, dtype=np.int32)
        self.period_confidence = np.asarray(confidences, dtype=np.float32)

    def _analyze_periodicity(self, qpos: np.ndarray) -> Dict[str, Union[bool, int, float]]:
        """
        Estimate whether a trajectory is periodic and derive repeat/cycle stats.
        Falls back to non-periodic (repeat=1) when heuristics are inconclusive.
        """
        hip_idx = self.jnt_name.get('hip_flexion_r', 6)
        sig = np.asarray(qpos[hip_idx], dtype=np.float64)
        idxs = [
            self.jnt_name['pelvis_tx'],
            self.jnt_name['pelvis_ty'],
            self.jnt_name['pelvis_tz'],
        ]
        coords = np.asarray(qpos[idxs], dtype=np.float64)

        T = sig.shape[0]
        if T <= 1:
            return {'periodic': False, 'repeat': 1, 'cycle_frames': max(int(T), 1), 'confidence': 0.0}

        sig_centered = sig - sig.mean()
        ac = np.correlate(sig_centered, sig_centered, mode='full')
        ac = ac[ac.size // 2 :]
        energy = float(ac[0])
        ac[0] = 0.0

        min_lag = max(int(0.3 * self.sample_frequency), 1)
        max_lag = min(int(2.0 * self.sample_frequency), T - 1)
        if max_lag <= min_lag or energy <= 0.0:
            return {'periodic': False, 'repeat': 1, 'cycle_frames': T, 'confidence': 0.0}

        window = ac[min_lag:max_lag]
        if window.size == 0:
            return {'periodic': False, 'repeat': 1, 'cycle_frames': T, 'confidence': 0.0}

        idx = int(np.argmax(window))
        peak = float(window[idx])
        lag = idx + min_lag
        norm_peak = peak / energy if energy > 0.0 else 0.0

        ranges = np.ptp(coords, axis=1)
        vertical_axis = int(np.argmin(ranges))
        horiz_axes = [ax for ax in range(3) if ax != vertical_axis]
        if len(horiz_axes) != 2:
            horiz_axes = [0, 2]

        horizontal = coords[horiz_axes]
        delta = np.diff(horizontal, axis=1)
        step_d = np.hypot(delta[0], delta[1])

        active = step_d > self._STEP_ACTIVITY_EPS
        if np.any(active):
            dirs = np.arctan2(delta[1, active], delta[0, active])
            dirs = np.unwrap(dirs)
            heading_span = float(dirs.max() - dirs.min()) if dirs.size else 0.0
        else:
            heading_span = 0.0

        repeat = max(int(round(T / lag)), 1)
        cycle_frames = max(int(round(T / repeat)), 1) if repeat > 0 else T

        valid_cycle = (
            norm_peak >= self._AC_PEAK_THRESHOLD
            and heading_span <= self._HEADING_SPAN_THRESHOLD
            and lag >= self._MIN_REPEAT_FRAMES
            and lag < T
        )

        if not valid_cycle:
            return {'periodic': False, 'repeat': 1, 'cycle_frames': T, 'confidence': norm_peak}

        cycle_frames = max(lag, cycle_frames, 1)
        cycle_frames = min(cycle_frames, T)

        return {
            'periodic': True,
            'repeat': max(repeat, 1),
            'cycle_frames': cycle_frames,
            'confidence': norm_peak,
        }

    def _estimate_mean_speed(self, qpos: np.ndarray) -> float:
        """
        Estimate the mean scalar speed of the pelvis trajectory projected onto
        the ground plane. Ignores initial/terminal idle segments and suppresses
        tiny jitters to remain robust for turning or looping motions.
        """
        idxs = [
            self.jnt_name['pelvis_tx'],
            self.jnt_name['pelvis_ty'],
            self.jnt_name['pelvis_tz'],
        ]
        coords = np.asarray(qpos[idxs], dtype=np.float64, order='C')  # (3, T)
        T = coords.shape[1]
        if T <= 1:
            return 0.0

        ranges = np.ptp(coords, axis=1)
        vertical_axis = int(np.argmin(ranges))
        horiz_axes = [ax for ax in range(3) if ax != vertical_axis]
        if len(horiz_axes) != 2:
            horiz_axes = [0, 2]

        dx = np.diff(coords[horiz_axes[0]], axis=0)
        dz = np.diff(coords[horiz_axes[1]], axis=0)
        step_d = np.hypot(dx, dz)
        if step_d.size == 0:
            return 0.0

        active = step_d > self._STEP_ACTIVITY_EPS
        if not np.any(active):
            return 0.0

        active_idx = np.flatnonzero(active)
        start = int(active_idx[0])
        stop = int(active_idx[-1]) + 1  # last active frame index

        path_len = float(step_d[start:stop].sum())

        duration = max((stop - start) / self.sample_frequency, 1e-6)
        speed = path_len / duration
        return float(max(speed, 0.0))

    def __len__(self) -> int:
        """Return number of trajectories available."""
        return len(self.qpos_list)
    
    def length(self) -> int:
        return len(self.qpos_list)

    def get(self, traj_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve (qpos, qvel) for a given trajectory index.

        Args:
            traj_id: Index of trajectory in [0, len).

        Returns:
            Tuple of (qpos, qvel), each of shape (52, T).

        Raises:
            IndexError: If traj_id is out of range.
        """
        if not 0 <= traj_id < len(self):
            raise IndexError(f"traj_id {traj_id} out of range")
        return self.qpos_list[traj_id], self.qvel_list[traj_id]

    def get_speed(self, traj_id: int) -> float:
        """
        Get average forward speed for trajectory.

        Args:
            traj_id: Index of trajectory.

        Returns:
            Speed in meters per second.

        Raises:
            IndexError: If traj_id is out of range.
        """
        if not 0 <= traj_id < len(self):
            raise IndexError(f"traj_id {traj_id} out of range")
        return float(self.speeds_list[traj_id])

    def get_foot_start(self, traj_id: int) -> int:
        """
        Get foot-start indicator for trajectory.

        Args:
            traj_id: Index of trajectory.

        Returns:
            1 if left foot starts first, 0 if right.

        Raises:
            IndexError: If traj_id is out of range.
        """
        if not 0 <= traj_id < len(self):
            raise IndexError(f"traj_id {traj_id} out of range")
        return int(self.foot_starts[traj_id])

    def is_periodic(self, traj_id: int) -> bool:
        if not 0 <= traj_id < len(self):
            raise IndexError(f"traj_id {traj_id} out of range")
        return bool(self.periodic_flags[traj_id])

    def get_repeat_count(self, traj_id: int) -> int:
        if not 0 <= traj_id < len(self):
            raise IndexError(f"traj_id {traj_id} out of range")
        return int(self.repeat_counts[traj_id])

    def get_cycle_frames(self, traj_id: int) -> int:
        if not 0 <= traj_id < len(self):
            raise IndexError(f"traj_id {traj_id} out of range")
        return int(self.cycle_lengths[traj_id])

    def get_periodicity_confidence(self, traj_id: int) -> float:
        if not 0 <= traj_id < len(self):
            raise IndexError(f"traj_id {traj_id} out of range")
        return float(self.period_confidence[traj_id])

    @property
    def all_speeds(self) -> List[float]:
        """List of average speeds for all trajectories."""
        return self.speeds_list.tolist()

    @property
    def all_foot_starts(self) -> List[int]:
        """List of foot-start flags for all trajectories."""
        return self.foot_starts.tolist()

    @property
    def all_periodic_flags(self) -> List[bool]:
        return self.periodic_flags.tolist()

    @property
    def all_repeat_counts(self) -> List[int]:
        return self.repeat_counts.tolist()

    def get_jnt_map(self) -> Dict[str, int]:
        """
        Get a copy of the joint name → index mapping.

        Returns:
            A dict mapping joint names to their index in qpos/qvel.
        """
        return dict(self.jnt_name)

    def get_sample_frequency(self) -> float:
        """
        Return the sampling frequency used for speed computation.

        Returns:
            Sampling rate in Hz.
        """
        return self.sample_frequency
    

class ReferenceTrajectories:
    """
    Per-environment cursor into a shared set of 52-DOF trajectories provided
    by TrajectoryManager.

    On reset, selects (or keeps) a trajectory, sets phase, and initializes
    the frame pointer. Each call to step() returns the reference (qpos, qvel)
    for the current frame, advancing by `increment` frames (with optional holds).

    Attributes
    ----------
    manager : TrajectoryManager
        Shared trajectories provider.
    traj_id : int
        Index of the current trajectory.
    qpos, qvel : np.ndarray
        Arrays of shape (52, T) for the selected trajectory (read-only views).
    traj_frames : int
        Total number of frames in the trajectory.
    periodic : bool
        True if the current clip is periodic; False for single-pass trajectories.
    repeat_times : int
        Estimated repeat count supplied by TrajectoryManager (>=1).
    step_frames : int
        Frames per repeat cycle; equals traj_frames when non-periodic.
    speed : float
        Average speed supplied by TrajectoryManager.
    foot_start : int
        1 if left foot starts first, else 0.
    _pos : int
        Current frame index.
    _hold_counter : int
        Number of consecutive holds at the current frame.
    _has_reached_end : bool
        True once the cursor has reached the final frame.
    """

    __slots__ = (
        'logger',
        'manager',
        'n_traj',
        'increment',
        'max_hold_steps',
        'rng',
        'jnt_name',
        'sample_frequency',
        'traj_id',
        'qpos',
        'qvel',
        'traj_frames',
        'periodic',
        'repeat_times',
        'step_frames',
        'speed',
        'foot_start',
        '_pos',
        '_hold_counter',
        '_has_reached_end',
    )

    def __init__(
        self,
        manager: TrajectoryManager,
        traj_id: Optional[int] = None,
        increment: int = 2,
        max_hold_steps: int = 5,
        random_seed: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Bind to a TrajectoryManager and initialize internal state.

        Parameters
        ----------
        manager : TrajectoryManager
            Shared trajectory loader.
        traj_id : Optional[int], default=None
            Specific trajectory to select; if None, chosen randomly.
        increment : int, default=2
            Number of frames to advance on each call to step().
        max_hold_steps : int, default=5
            Maximum consecutive holds before advancing.
        random_seed : Optional[int], default=None
            Seed for reproducible random selection.
        verbose : bool, default=False
            Enable debug logging if True.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        self.manager = manager
        self.n_traj = manager.length() if hasattr(manager, "length") else len(manager)
        self.increment = int(increment)
        self.max_hold_steps = int(max_hold_steps)
        self.rng = np.random.default_rng(random_seed)
        self.traj_id = -1
        self.qpos = None
        self.qvel = None
        self.traj_frames = 0
        self.periodic = False
        self.repeat_times = 1
        self.step_frames = 1
        self.speed = 0.0
        self.foot_start = 0
        self._pos = 0
        self._hold_counter = 0
        self._has_reached_end = False

        # fixed joint mapping and sampling rate
        self.jnt_name = manager.get_jnt_map()
        self.sample_frequency = manager.get_sample_frequency()

        # pick and load trajectory
        self.set_trajectory(traj_id)

    def set_trajectory(self, traj_id: Optional[int] = None) -> None:
        """
        Switch to the specified trajectory, or random if None,
        and reset frame pointer and hold counter.

        Parameters
        ----------
        traj_id : Optional[int]
            Index of trajectory to select.
        """
        if traj_id is None:
            self.traj_id = int(self.rng.integers(self.n_traj))
        else:
            if not (0 <= traj_id < self.n_traj):
                raise IndexError(f"traj_id {traj_id} out of range [0, {self.n_traj-1}]")
            self.traj_id = traj_id

        # load qpos and qvel (read‐only views)
        self.qpos, self.qvel = self.manager.get(self.traj_id)
        self.qpos.setflags(write=False)
        self.qvel.setflags(write=False)
        assert not self.qpos.flags.writeable and not self.qvel.flags.writeable, \
            "Underlying trajectory data must be read-only"
            
        self.traj_frames = self.qpos.shape[1]
        self.periodic = self.manager.is_periodic(self.traj_id)
        repeat_est = self.manager.get_repeat_count(self.traj_id)
        cycle_est = self.manager.get_cycle_frames(self.traj_id)
        if self.periodic:
            self.repeat_times = max(repeat_est, 1)
            self.step_frames = max(min(cycle_est, self.traj_frames), 1)
        else:
            self.repeat_times = 1
            self.step_frames = self.traj_frames
        self.speed = self.manager.get_speed(self.traj_id)
        self.foot_start = self.manager.get_foot_start(self.traj_id)

        # initialize cursor state
        self._pos = 0
        self._hold_counter = 0
        self._has_reached_end = False

        self.logger.debug(
            f"Switched to trajectory {self.traj_id}: "
            f"frames={self.traj_frames}, cycle={self.step_frames}, "
            f"speed={self.speed:.2f}, foot_start={'left' if self.foot_start else 'right'}"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        phase: Optional[float] = None,
        randomize_traj: bool = True,
        traj_id: Optional[int] = None
        ) -> None:
        """
        Reinitialize this ReferenceTrajectories cursor.

        Parameters
        ----------
        seed : Optional[int]
            If provided, reseed the internal RNG (affects random trajectory/phase).
        phase : Optional[float]
            If provided (0-100), sets exact gait phase; if None, choose random [0,100).
        randomize_traj : bool
            If True and traj_id is None, pick a random trajectory; else keep current or use provided.
        traj_id : Optional[int]
            If provided, switch to this exact trajectory; else behavior depends on randomize_traj.
    
        Raises
        ------
        TypeError
            If 'phase' or 'traj_id' have invalid types.
        ValueError
            If 'phase' is out of [0,100].
        IndexError
            If 'traj_id' is out of range.
        """
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError(f"seed must be int or None, got {type(seed)}")
            self.rng = np.random.default_rng(seed)
        
        if randomize_traj:
            self.set_trajectory(traj_id)
        elif traj_id is not None:
            self.set_trajectory(traj_id)

        if phase is None:
            phase_val = float(self.rng.uniform(0.0, 100.0))
        else:
            if not isinstance(phase, (float, int)):
                raise TypeError(f"phase must be float or None, got {type(phase)}")
            phase_val = float(phase)
            if not (0.0 <= phase_val <= 100.0):
                raise ValueError("phase must be within [0,100]")

        if self.step_frames > 1:
            idx = int(round(phase_val / 100.0 * (self.step_frames - 1)))
        else:
            idx = 0
            
        self._pos = idx
        self._hold_counter = 0
        self._has_reached_end = (self._pos >= self.traj_frames - 1)
        self.logger.debug(
            f"reset(): traj_id={self.traj_id}, phase={phase_val:.2f}% -> frame={self._pos}, "
            f"has_reached_end={self._has_reached_end}"
        )

    def next(self, hold_phase: bool = False) -> None:
        """
        Advance the frame pointer by 'increment', or hold in place up to 'max_hold_steps'.
    
        Parameters
        ----------
        hold_phase : bool
            If True, delay advancement for at most 'max_hold_steps' calls; otherwise reset hold counter.
        """
        if hold_phase:
            self._hold_counter += 1
            if self._hold_counter < self.max_hold_steps:
                self.logger.debug(f"Holding at frame {self._pos} (hold count {self._hold_counter})")
                return
            
            self.logger.debug(f"Reached max hold ({self.max_hold_steps}) at frame {self._pos}, advancing")
            self._hold_counter = 0
        else:
            if self._hold_counter:
                self.logger.debug(f"Hold counter reset from {self._hold_counter} to 0")
            self._hold_counter = 0
        
        if self._has_reached_end:
            return
        
        self._pos += self.increment
        
        if self._pos >= self.traj_frames - 1:
            self._pos = self.traj_frames - 1
            self._has_reached_end = True
            self.logger.debug(f"Reached end of trajectory at frame {self._pos}")
        else:
            self.logger.debug(f"Advanced to frame {self._pos}")

    @property
    def phase(self) -> float:
        """
        Current phase percentage [0,100] within one cycle.
        """
        if self.step_frames <= 1:
            return 0.0
        cycle_idx = self._pos % self.step_frames
        return cycle_idx / (self.step_frames - 1) * 100.0

    @property
    def has_reached_end(self) -> bool:
        """
        True if the cursor is at the final frame of the trajectory.
        """
        return self._has_reached_end
    
    def _idx_clip(self, idx: int) -> int:
        T = self.traj_frames
        if idx < 0: 
            return 0
        if idx >= T: 
            return T - 1
        return idx
    
    def peek(self, offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, 'qpos') or not hasattr(self, 'qvel'):
            raise RuntimeError("Trajectory not initialized; call set_trajectory() first.")
        i = self._idx_clip(int(self._pos) + int(offset))
        return self.qpos[:, i], self.qvel[:, i]

    def get_qpos(self) -> np.ndarray:
        """
        Return the qpos vector at the current frame.
        """
        if not hasattr(self, 'qpos'):
            raise RuntimeError("Trajectory not initialized; call set_trajectory() first.")
        return self.qpos[:, self._pos]

    def get_qvel(self) -> np.ndarray:
        """
        Return the qvel vector at the current frame.
        """
        if not hasattr(self, 'qvel'):
            raise RuntimeError("Trajectory not initialized; call set_trajectory() first.")
        return self.qvel[:, self._pos]

    def get_reference_trajectories(self, offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        return self.peek(offset)
    
    def get_reference_window(
            self,
            start_steps: int = 0,
            num_steps: int = 1,
            *,
            step_major: bool = True,
            ) -> Tuple[np.ndarray, np.ndarray]:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not hasattr(self, 'qpos') or not hasattr(self, 'qvel'):
            raise RuntimeError("Trajectory not initialized; call set_trajectory() first.")
    
        T = self.traj_frames
        Dq = int(self.qpos.shape[0])
        Dv = int(self.qvel.shape[0])
        inc   = int(getattr(self, "increment", 1))
        
        start_frame = int(self._pos) + int(start_steps) * inc
        last_frame  = start_frame + (num_steps - 1) * inc
    
        if start_frame >= 0 and last_frame <= T - 1:
            qpos_blk = self.qpos[:, start_frame : last_frame + 1 : inc]  # (D, S) view
            qvel_blk = self.qvel[:, start_frame : last_frame + 1 : inc]
            return (qpos_blk.T, qvel_blk.T) if step_major else (qpos_blk, qvel_blk)
    
        S = int(num_steps)
        Q = np.empty((Dq, S), dtype=self.qpos.dtype)
        V = np.empty((Dv, S), dtype=self.qvel.dtype)
    
        for k in range(S):
            idx = start_frame + k * inc
            if idx < 0:
                use = 0
            elif idx >= T:
                use = T - 1
            else:
                use = idx
            Q[:, k] = self.qpos[:, use]
            V[:, k] = self.qvel[:, use]

        return (Q.T, V.T) if step_major else (Q, V)

    def get_pelvis_ang(self) -> np.ndarray:
        """
        Extract pelvis [tz, ty, tx, tilt, list, rotation] at current frame.
        """
        keys = ['pelvis_tz','pelvis_ty','pelvis_tx',
                'pelvis_tilt','pelvis_list','pelvis_rotation']
        idxs = [self.jnt_name[k] for k in keys]
        return self.qpos[idxs, self._pos]

    def get_pelvis_angV(self) -> np.ndarray:
        """
        Extract pelvis velocities at current frame.
        """
        keys = ['pelvis_tz','pelvis_ty','pelvis_tx',
                'pelvis_tilt','pelvis_list','pelvis_rotation']
        idxs = [self.jnt_name[k] for k in keys]
        return self.qvel[idxs, self._pos]

    def get_torso_ang(self) -> np.ndarray:
        """
        Extract torso [lumbar_extension, lumbar_bending, lumbar_rotation].
        """
        keys = ['lumbar_extension','lumbar_bending','lumbar_rotation']
        idxs = [self.jnt_name[k] for k in keys]
        return self.qpos[idxs, self._pos]

    def get_torso_angV(self) -> np.ndarray:
        """
        Extract torso angular velocities at current frame.
        """
        keys = ['lumbar_extension','lumbar_bending','lumbar_rotation']
        idxs = [self.jnt_name[k] for k in keys]
        return self.qvel[idxs, self._pos]

    def get_joint_data(
        self,
        joint_group: Union[str, List[str]],
        data_type: str = "angle"
        ) -> np.ndarray:
        """
        Retrieve reference trajectory data (position or velocity) for specified joint(s) at the current frame.
    
        Parameters
        ----------
        joint_group : Union[str, List[str]]
            Either a predefined group ('pelvis', 'lumbar', 'thorax'), a single joint
            name, or a list of joint names.
        data_type : str, optional
            Type of data to return:
              - 'angle' (qpos) for joint positions.
              - 'velocity' (qvel) for joint angular velocities.
            Default is 'angle'.
    
        Returns
        -------
        np.ndarray
            1D array of length N (number of joints requested), containing the data at the current frame.
    
        Raises
        ------
        ValueError
            If an unknown joint name is requested or if `data_type` is not 'angle'/'velocity'.
        """
        joint_map = {
            "pelvis": ['pelvis_tz', 'pelvis_ty', 'pelvis_tx', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
            "lumbar": ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation'],
            "thorax": ['thorax_extension', 'thorax_bending', 'thorax_rotation']
        }
        # Determine the list of joint names
        if isinstance(joint_group, str):
            if joint_group in joint_map:
                names = joint_map[joint_group]
            else:
                names = [joint_group]
        elif isinstance(joint_group, list):
            names = joint_group
        else:
            raise ValueError("joint_group must be a str or a list of str")
        indices: List[int] = []
        for name in names:
            if name not in self.jnt_name:
                raise ValueError(f"Joint name '{name}' not found in mapping.")
            indices.append(self.jnt_name[name])
        # Extract the corresponding data
        if data_type == "angle":
            if self.qpos is None:
                raise RuntimeError("qpos data is not initialized.")
            return self.qpos[indices, self._pos]
        elif data_type == "velocity":
            if self.qvel is None:
                raise RuntimeError("qvel data is not initialized.")
            return self.qvel[indices, self._pos]
        else:
            raise ValueError("data_type must be 'angle' or 'velocity'.")
