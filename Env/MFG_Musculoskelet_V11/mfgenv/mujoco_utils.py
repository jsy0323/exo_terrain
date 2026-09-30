"""
Utilities for loading and managing MuJoCo models.
@author: YAKE
"""

import os
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Configure module-level logger
logger = logging.getLogger(__name__)

if not logger.handlers:
    # Add a default StreamHandler if no handler exists.
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def load_mujoco_model(
        model_path: str,
        use_muscle: bool = False,
        load_mesh: bool = False,
        ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Load a MuJoCo model from XML, removing either:
      - all group=2 <position> actuators (if use_muscle=True), or
      - all group=3 <general class="ZXYmuscle"> actuators (if use_muscle=False).

    Parameters
    ----------
    model_path : str
        Path to the MuJoCo XML model file.
    use_muscle : bool, default=False
        If False: keep group=1 & group=2 (position actuators), drop group=3 (muscles).
        If True:  keep group=1 & group=3 (muscles), drop group=2 (low_position).

    Returns
    -------
    model : mujoco.MjModel
        The filtered MuJoCo model.
    data : mujoco.MjData
        Simulation data initialized for the model.

    Raises
    ------
    FileNotFoundError
        If model_path does not exist.
    ValueError
        If the filtered model has zero joints or bodies.
    RuntimeError
        If parsing or loading fails.

    Notes
    -----
    The path is expanded to an absolute path before loading. The XML is parsed via
    :class:`mujoco.MjSpec` and modified in-place to drop undesired actuator groups.
    """
    abs_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"MuJoCo model file not found: {abs_path}")

    try:
        spec = mujoco.MjSpec.from_file(abs_path)

        keep_groups = {0, 2} if use_muscle else {0, 1}
        for actuator in tuple(spec.actuators):
            try:
                group_id = int(actuator.group)
            except Exception:
                continue
            if group_id not in keep_groups:
                actuator.delete()

        spec.compiler.discardvisual = 0 if load_mesh else 1

        model = spec.compile()
        if model is None:
            raise ValueError("Compiled model is None. Check the XML file for syntax errors.")

        data = mujoco.MjData(model)

        if model.njnt == 0:
            raise ValueError("Model contains no joints. Ensure the XML file defines joints correctly.")
        if model.nbody == 0:
            raise ValueError("Model contains no bodies. Ensure at least one rigid body is defined.")

        logger.debug("Successfully loaded MuJoCo model from %s", abs_path)
        return model, data

    except mujoco.FatalError as e:
        raise RuntimeError(f"MuJoCo Fatal Error: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load MuJoCo model from {abs_path}: {e}")
        
def parse_actuator_prm_from_xml(model_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse actuator parameters from a MuJoCo XML model file, handling both position
    and muscle actuators with minimal intermediate allocations.

    Returns a dict mapping actuator name → dict of parameters:
      - For position actuators (class="low_position" or "up_position"):
          { "type": "position",
            "kp": float,
            "forcerange": [float, float],
            "limited": bool,
            "range": [float, float] }

      - For muscle actuators (class="ZXYmuscle"):
          { "type": "muscle",
            "tendon": str,
            "ctrlrange": [float, float],
            "lengthrange": [float, float],
            "gainprm": List[float],
            "biasprm": List[float] }

    If a muscle <general> element lacks a 'ctrlrange', this function falls back
    to the <default class="ZXYmuscle"><general ctrlrange=... /></default> definition.

    Parameters
    ----------
    model_path : str
        Path to the MuJoCo XML model file.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping from actuator name to parameter dict as described above.

    Raises
    ------
    FileNotFoundError
        If `model_path` does not exist.
    ValueError
        If required attributes are missing or invalid.
    ET.ParseError
        If the XML cannot be parsed.

    Notes
    -----
    The function performs minimal intermediate allocations by reusing helper
    parsing routines and iterating the XML tree lazily.
    """
    abs_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"MuJoCo XML model file not found: {abs_path}")

    try:
        tree = ET.parse(abs_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file: {e}")

    actuator_params: Dict[str, Dict[str, Any]] = {}

    # Locate default <general> for ZXYmuscle (to obtain fallback ctrlrange if missing)
    default_muscle_elem = root.find(".//default[@class='ZXYmuscle']/general")
    default_ctrlrange = None
    if default_muscle_elem is not None:
        ctrl_str = default_muscle_elem.get("ctrlrange")
        if ctrl_str:
            try:
                default_ctrlrange = list(map(float, ctrl_str.split()))
                if len(default_ctrlrange) != 2:
                    default_ctrlrange = None
            except ValueError:
                default_ctrlrange = None

    def _parse_float_pair(value: Optional[str], name: str) -> List[float]:
        if value is None:
            raise ValueError(f"Missing '{name}' attribute.")
        try:
            vals = list(map(float, value.split()))
        except ValueError:
            raise ValueError(f"Invalid '{name}': '{value}'")
        if len(vals) != 2:
            raise ValueError(f"'{name}' must contain exactly two floats, got {value}")
        return vals

    # 1) Parse <actuator><position .../> elements
    for pos in root.iterfind(".//actuator/position"):
        act_name = pos.get("name")
        if not act_name:
            raise ValueError("A <position> actuator is missing the 'name' attribute.")

        joint_name = pos.get("joint")
        if not joint_name:
            raise ValueError(f"Actuator '{act_name}' missing 'joint' attribute.")

        kp_str = pos.get("kp")
        if kp_str is None:
            raise ValueError(f"Actuator '{act_name}' missing 'kp' attribute.")
        try:
            kp_val = float(kp_str)
        except ValueError:
            raise ValueError(f"Invalid kp for actuator '{act_name}': '{kp_str}'")

        joint_elem = root.find(f".//joint[@name='{joint_name}']")
        if joint_elem is None:
            raise ValueError(f"No <joint name='{joint_name}'> found for actuator '{act_name}'.")

        afr_vals = _parse_float_pair(
            joint_elem.get("actuatorfrcrange"),
            f"actuatorfrcrange for joint '{joint_name}' of actuator '{act_name}'"
        )

        limited_str = joint_elem.get("limited", "false")
        limited = limited_str.strip().lower() == "true"

        rng_vals = _parse_float_pair(
            joint_elem.get("range", "-1 1"),
            f"range for joint '{joint_name}'"
        )

        actuator_params[act_name] = {
            "type": "position",
            "kp": kp_val,
            "forcerange": afr_vals,
            "limited": limited,
            "range": rng_vals
        }

    # 2) Parse <actuator><general .../> elements of class="ZXYmuscle"
    for gen in root.iterfind(".//actuator/general"):
        cls = gen.get("class", "")
        if cls.strip() != "ZXYmuscle":
            continue

        act_name = gen.get("name")
        if not act_name:
            raise ValueError("A <general> actuator of class 'ZXYmuscle' is missing 'name'.")

        tendon_name = gen.get("tendon")
        if not tendon_name:
            raise ValueError(f"Muscle actuator '{act_name}' missing 'tendon' attribute.")

        # ctrlrange: if absent, fallback to default_muscle_elem
        ctrl_str = gen.get("ctrlrange")
        if ctrl_str is None:
            if default_ctrlrange is None:
                raise ValueError(f"Muscle actuator '{act_name}' missing 'ctrlrange', and no default found.")
            ctrl_vals = default_ctrlrange
        else:
            try:
                ctrl_vals = list(map(float, ctrl_str.split()))
                if len(ctrl_vals) != 2:
                    raise ValueError
            except ValueError:
                raise ValueError(f"Invalid 'ctrlrange' for muscle '{act_name}': '{ctrl_str}'")

        # lengthrange
        length_vals = _parse_float_pair(
            gen.get("lengthrange"),
            f"lengthrange for muscle '{act_name}'"
        )

        # gainprm
        gain_str = gen.get("gainprm")
        if gain_str is None:
            raise ValueError(f"Muscle actuator '{act_name}' missing 'gainprm'.")
        try:
            gain_vals = list(map(float, gain_str.split()))
        except ValueError:
            raise ValueError(f"Invalid 'gainprm' for muscle '{act_name}': '{gain_str}'")

        # biasprm
        bias_str = gen.get("biasprm")
        if bias_str is None:
            raise ValueError(f"Muscle actuator '{act_name}' missing 'biasprm'.")
        try:
            bias_vals = list(map(float, bias_str.split()))
        except ValueError:
            raise ValueError(f"Invalid 'biasprm' for muscle '{act_name}': '{bias_str}'")

        actuator_params[act_name] = {
            "type": "muscle",
            "tendon": tendon_name,
            "ctrlrange": ctrl_vals,
            "lengthrange": length_vals,
            "gainprm": gain_vals,
            "biasprm": bias_vals
        }

    return actuator_params
        
def check_invalid_names(
    jnt_names: List[str],
    actuator_names: List[str],
    use_muscle: bool = False,
    verbose: bool = False
    ) -> None:
    """
    Validate consistency between joint names and actuator names.

    Specifically:
      1) Every position actuator name must correspond to a joint name.
      2) When `use_muscle` is False, the number of undriven joints (free-joint DoFs)
         must be either 1 (floating pelvis) or 7 (full free joint).

    Muscle actuators, whose names do not appear in `jnt_names`, are excluded from
    joint-name validation. Duplicate position actuator names are disallowed.

    Parameters
    ----------
    jnt_names : List[str]
        Names of all joints in the model.
    actuator_names : List[str]
        Names of all actuators (position + muscle). Position actuators are assumed
        to share a name with their driven joint.
    use_muscle : bool, default=False
        If True, skip the free-joint DoF count check.
    verbose : bool, default=False
        If True, emit logging information about the partition.

    Raises
    ------
    ValueError
        If a position actuator does not correspond to any joint, or (when
        `use_muscle=False`) the undriven joint count is not 1 or 7.
    """
    jnt_set = set(jnt_names)
    pos_acts = [name for name in actuator_names if name in jnt_set]
    muscle_acts = [name for name in actuator_names if name not in jnt_set]

    if len(pos_acts) != len(set(pos_acts)):
        from collections import Counter
        duplicates = [name for name, count in Counter(pos_acts).items() if count > 1]
        raise ValueError(f"Duplicate position actuator names detected: {sorted(duplicates)}")

    # All position actuator names must appear in joint set
    missing = [name for name in pos_acts if name not in jnt_set]
    if missing:
        raise ValueError(f"Position actuators missing in joint list: {missing}")

    extra_joints: Optional[List[str]] = None
    if not use_muscle:
        driven = set(pos_acts)
        extra_joints = sorted(jnt_set - driven)
        if len(extra_joints) not in (1, 7):
            raise ValueError(
                f"Expected 1 or 7 free-joint DoFs, but found {len(extra_joints)}: {extra_joints}"
            )

    if verbose:
        logger.info("Position actuators: %s", pos_acts)
        logger.info("Muscle actuators (skipped): %s", muscle_acts)
        if extra_joints is not None:
            logger.info("Extra joints (free-joint DoFs): %s", extra_joints)

def compute_muscle_volumes(
    actuator_prms: Dict[str, Dict],
    actuator_names: List[str],
    specific_tension: float = 2.68e5
    ) -> np.ndarray:
    """
    Compute muscle volumes from pre-parsed actuator parameters.

    Optimal fiber length (l_opt) is computed as:
        l_opt = (lengthrange[1] - lengthrange[0]) / (gainprm[1] - gainprm[0])
    Physiological cross-sectional area (PCSA):
        PCSA = F_max / specific_tension
    Muscle volume (V):
        V = PCSA * l_opt

    Args:
        actuator_prms (Dict[str, Dict]): Mapping actuator name -> parameter dict,
            including 'lengthrange' and 'gainprm' for muscle actuators.
        actuator_names (List[str]): Ordered list of actuator names (env.actuator_names).
        specific_tension (float): Muscle specific tension (N/m^2), default 3e5.

    Returns
    -------
    np.ndarray
        Array of volumes with shape (len(actuator_names),), aligned with actuator_names.

    Notes
    -----
    If ``gainprm[1] == gainprm[0]`` for a muscle actuator, the volume is set to 0 and a
    warning is logged to avoid division by zero.
    """
    volumes = np.zeros(len(actuator_names), dtype=np.float64)
    inv_tension = 1.0 / specific_tension

    for idx, name in enumerate(actuator_names):
        prm = actuator_prms.get(name)
        if prm is None or prm.get('type') != 'muscle':
            continue
        lr = prm['lengthrange']
        gp = prm['gainprm']
        denom = gp[1] - gp[0]
        if abs(denom) < 1e-12:
            logger.warning(
                "Gain parameters for muscle '%s' yield near-zero denominator; volume set to 0.",
                name
            )
            continue
        l_opt = (lr[1] - lr[0]) / denom
        volumes[idx] = gp[2] * inv_tension * l_opt

    return volumes
        
def check_and_enable_Limit(
        model: mujoco.MjModel, 
        jnt_names: List[str], 
        actuator_names: List[str], 
        actuator_prm: Dict[str, Any], 
        default_forcerange: Optional[Sequence[float]] = None
        ) -> None:
    """
    Enable force and control limits for actuators and joints in the MuJoCo model.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model to be modified in-place.
    jnt_names : List[str]
        Joint names ordered as in the model.
    actuator_names : List[str]
        Actuator names ordered as in the model (position + muscle).
    actuator_prm : dict
        Actuator parameter mapping produced by :func:`parse_actuator_prm_from_xml`.
    default_forcerange : sequence, optional
        Fallback joint force range ``[min, max]`` used when an entry is missing.

    Raises
    ------
    ValueError
        If actuator parameters are invalid or arguments have incorrect types.

    Notes
    -----
    This function updates ``model.jnt_actfrclimited``, ``model.jnt_actfrcrange``,
    and for matching actuators also ``model.actuator_ctrllimited`` and
    ``model.actuator_ctrlrange``. Pelvis/free joints (names containing
    ``floating`` or ``beta``) are skipped.
    """
    if not isinstance(jnt_names, list):
        raise ValueError("jnt_names must be a list.")
    if not isinstance(actuator_names, list):
        raise ValueError("actuator_names must be a list.")
    if not isinstance(actuator_prm, dict):
        raise ValueError("actuator_prm must be a dictionary.")
    
    act_index = {name: idx for idx, name in enumerate(actuator_names)}
    if default_forcerange is None:
        default_force = np.array([-100.0, 100.0], dtype=np.float64)
    else:
        default_force = np.asarray(default_forcerange, dtype=np.float64)
        if default_force.shape != (2,):
            raise ValueError("default_forcerange must be a sequence of two floats.")
    
    for i, jnt_name in enumerate(jnt_names):
        # Skip pelvis joints
        if 'floating' in jnt_name.lower() or 'beta' in jnt_name.lower():
            continue
        model.jnt_actfrclimited[i] = 1  # Enable joint force limits
        
        actuator_idx = act_index.get(jnt_name)
        if actuator_idx is not None:
            if model.actuator_ctrllimited[actuator_idx] != 1:
                model.actuator_ctrllimited[actuator_idx] = 1
            ctrl_range = actuator_prm[jnt_name]['range']
            if not np.array_equal(model.actuator_ctrlrange[actuator_idx], ctrl_range):
                model.actuator_ctrlrange[actuator_idx] = ctrl_range
    
        if jnt_name in actuator_prm:
            forcerange = actuator_prm[jnt_name].get("forcerange")
            if not isinstance(forcerange, (list, tuple)) or len(forcerange) != 2:
                raise ValueError(f"Invalid forcerange for actuator '{jnt_name}': {forcerange}")
            model.jnt_actfrcrange[i] = [min(forcerange), max(forcerange)]
        else:
            model.jnt_actfrcrange[i] = default_force
    
    logger.info("Joint and actuator force limits successfully enabled.")
 
def reset_mujoco_state(
        model: mujoco.MjModel, 
        data: mujoco.MjData, 
        qpos_init: Optional[np.ndarray] = None, 
        qvel_init: Optional[np.ndarray] = None, 
        randomize: bool = False, 
        noise_std: float = 0.005,
        rng: Optional[np.random.Generator] = None
        ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Reset the MuJoCo simulation state to its initial conditions.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model.
    data : mujoco.MjData
        The simulation data (modified in-place).
    qpos_init : np.ndarray, optional
        Initial joint positions; must match ``model.nq``.
    qvel_init : np.ndarray, optional
        Initial joint velocities; must match ``model.nv``.
    randomize : bool, default=False
        If True, add Gaussian noise after assigning initial state.
    noise_std : float, default=0.005
        Standard deviation of the Gaussian noise.
    rng : np.random.Generator, optional
        Random generator used when ``randomize`` is True. If None, a generator is created.

    Returns
    -------
    Tuple[mujoco.MjModel, mujoco.MjData]
        The input ``(model, data)`` after resetting (returned for convenience).

    Raises
    ------
    ValueError
        If ``qpos_init`` or ``qvel_init`` shapes mismatch the model.
    RuntimeError
        If the resulting state contains NaN values.
    """
    mujoco.mj_resetData(model, data)
    
    if qpos_init is not None and qpos_init.shape[0] != model.nq:
        raise ValueError(f"qpos_init shape mismatch: expected {model.nq}, got {qpos_init.shape[0]}")
    if qvel_init is not None and qvel_init.shape[0] != model.nv:
        raise ValueError(f"qvel_init shape mismatch: expected {model.nv}, got {qvel_init.shape[0]}")
    
    if randomize and rng is None:
        rng = np.random.default_rng()

    if qpos_init is not None:
        data.qpos[:] = qpos_init
        if randomize:
            data.qpos += rng.normal(0.0, noise_std, size=data.qpos.shape)
        
    if qvel_init is not None:
        data.qvel[:] = qvel_init
        if randomize:
            data.qvel += rng.normal(0.0, noise_std, size=data.qvel.shape)
    
    mujoco.mj_forward(model, data)
    
    if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
        raise RuntimeError("Reset resulted in NaN values! The model initialization may be incorrect.")

    return model, data
    
def step_mujoco(
        model: mujoco.MjModel, 
        data: mujoco.MjData, 
        action: np.ndarray
        ) -> None:
    """
    Apply an action to the actuators and advance the simulation.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The simulation data.
        action (np.ndarray): Action vector applied to actuators.

    Raises
    ------
    ValueError
        If action shape is incorrect or values are out of bounds.
    RuntimeError
        If the simulation state contains NaN values.

    Notes
    -----
    Performs bounds checking against ``model.actuator_ctrlrange`` each call and logs
    a warning when actuator forces become unusually large (>1e4).
    """
    if action.shape[0] != model.nu:
        raise ValueError(f"Action shape mismatch: expected {model.nu}, got {action.shape[0]}")

    action_lower, action_upper = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    if np.any(action < action_lower) or np.any(action > action_upper):
        raise ValueError(f"Action out of bounds: expected range {model.actuator_ctrlrange}, got {action}")

    data.ctrl[:] = action
    mujoco.mj_step(model, data)

    if np.isnan(data.qpos).any() or np.isnan(data.qvel).any() or np.isnan(data.qacc).any():
        raise RuntimeError("Simulation state contains NaN values! The model may be unstable.")

    max_actuator_force = np.max(np.abs(data.qfrc_actuator))
    if max_actuator_force > 1e4:
        logger.warning("Unusually high actuator forces detected: %.2e. Consider adjusting model parameters.", max_actuator_force)

def sync_mujoco_with_state(
        model: mujoco.MjModel, 
        data: mujoco.MjData, 
        qpos: np.ndarray, 
        qvel: Optional[np.ndarray] = None
        ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Synchronize the simulation state with provided joint positions and velocities.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The simulation data.
        qpos (np.ndarray): New joint positions.
        qvel (np.ndarray, optional): New joint velocities.

    Returns
    -------
    Tuple[mujoco.MjModel, mujoco.MjData]
        The input ``(model, data)`` after synchronization.

    Raises
    ------
    ValueError
        If qpos or qvel dimensions do not match the model.
    RuntimeError
        If synchronization leads to NaN values.

    Notes
    -----
    If ``qvel`` is omitted, joint velocities are reset to zero.
    """
    if qpos.shape[0] != model.nq:
        raise ValueError(f"qpos shape mismatch: expected {model.nq}, got {qpos.shape[0]}")
    if qvel is not None and qvel.shape[0] != model.nv:
        raise ValueError(f"qvel shape mismatch: expected {model.nv}, got {qvel.shape[0]}")

    data.qpos[:] = qpos
    if qvel is None:
        data.qvel.fill(0.0)
    else:
        data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    
    if np.isnan(data.qpos).any() or np.isnan(data.qvel).any():
        raise RuntimeError("Synchronization resulted in NaN values! Check the input state.")

    return model, data
                    
