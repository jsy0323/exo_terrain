import gymnasium as gym
import numpy as np
import random
import logging
import time
from types import MappingProxyType
from typing import Optional, Tuple, Callable, Dict, Any, List, Mapping

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)

class BaseEnv(gym.Env):
    """
    Core Gymnasium base class consolidating RNG seeding and render lifecycle.

    Subclasses typically inherit from :class:`BaseEnv` → :class:`MuJoCoEnv`
    → task-specific environments.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        *,
        auto_render: bool = False,
        render_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        render_mode : {"human", "rgb_array"}, optional
            Active render mode. ``None`` disables rendering entirely.
        auto_render : bool, optional
            Whether :meth:`render` should produce visuals automatically.  Subclasses
            may toggle this flag during runtime (e.g. enable in evaluation only).
        render_options : dict, optional
            Optional immutable configuration exposed to render hooks. The mapping is
            stored in a read-only proxy and cannot be mutated after construction.
        """
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. "
                f"Valid options are {self.metadata['render_modes']} or None."
            )

        self.render_mode: Optional[str] = render_mode
        self.auto_render: bool = bool(auto_render)

        # Rendering backends (viewer/renderer) are created on demand by subclasses.
        self.viewer = None
        self.renderer = None
        self._last_human_render: Optional[float] = None

        # Immutable render configuration available to derived classes.
        self._render_options: Mapping[str, Any] = MappingProxyType(dict(render_options) if render_options else {})

        # Registered resource finalisers (executed in reverse order during close()).
        # Derived classes use this for ephemeral artefacts such as ghost MjData buffers.
        self._resource_stack: List[Callable[[], None]] = []

        # RNG placeholder – initialised lazily via self._seed().
        self._np_random: Optional[np.random.Generator] = None

        # Provide a per-instance logger for downstream classes.
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # RNG handling
    # ------------------------------------------------------------------
    def _seed(self, seed: Optional[int] = None) -> int:
        """
        Seed the environment RNGs.

        Subclasses should call this at the start of :meth:`reset` when a new seed
        is provided.
        """
        from gymnasium.utils import seeding

        self._np_random, actual = seeding.np_random(seed)
        u32 = int(actual) & 0xFFFFFFFF
        random.seed(u32)
        np.random.seed(u32)
        self.logger.debug("BaseEnv RNG seeded with %d", u32)
        return u32

    def seed(self, seed: Optional[int] = None):
        """
        Gymnasium-compatible public seeding API. Returns the seed inside a list.
        """
        return [self._seed(seed)]

    @property
    def np_random(self) -> np.random.Generator:
        """
        Lazily-initialised random number generator for subclasses.

        Accessing this property ensures a generator is available even if the user
        never called :meth:`seed`.  The generator uses the last seed provided via
        :meth:`seed`; if none is set, it seeds itself with ``None``.
        """
        if self._np_random is None:
            self._seed(None)
        return self._np_random

    # ------------------------------------------------------------------
    # Render configuration helpers
    # ------------------------------------------------------------------
    def set_render_mode(self, mode: Optional[str], *, auto_render: Optional[bool] = None) -> None:
        """
        Change the active render mode. Existing viewer/renderer handles are torn down.
        """
        if mode is not None and mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode: {mode}. "
                f"Valid options are {self.metadata['render_modes']} or None."
            )
        if mode == self.render_mode and auto_render is None:
            return

        self._destroy_render_backend()
        self.render_mode = mode
        if auto_render is not None:
            self.auto_render = bool(auto_render)

    def should_render(self) -> bool:
        """
        Whether :meth:`render` should produce output (subclasses may override).
        """
        return bool(self.auto_render)

    def set_auto_render(self, flag: bool) -> None:
        """Convenience setter for :attr:`auto_render`."""
        self.auto_render = bool(flag)

    def get_render_option(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a render option defined during initialisation.
        """
        return self._render_options.get(key, default)

    def get_render_options(self) -> Dict[str, Any]:
        """Return a shallow copy of all render options."""
        return dict(self._render_options)

    @property
    def render_options(self) -> Mapping[str, Any]:
        """Read-only mapping of configuration supplied at construction."""
        return self._render_options


    # Resource management -------------------------------------------------
    def register_resource(self, cleanup: Callable[[], None]) -> None:
        """
        Register a cleanup callable executed during :meth:`close`.

        Resources are executed in reverse order (LIFO).  The callable should raise no
        exception; any raised value is logged at debug level then ignored so that other
        cleanups still run.
        """
        if not callable(cleanup):
            raise TypeError("cleanup must be callable")
        self._resource_stack.append(cleanup)

    # ------------------------------------------------------------------
    # Rendering pipeline
    # ------------------------------------------------------------------
    def render(self, *, force: bool = False):
        """
        Render the environment once.

        Parameters
        ----------
        force : bool, optional
            Bypass :meth:`should_render`. Useful for manual inspection even when
            ``auto_render`` is disabled.

        Notes
        -----
        Rendering paths allocate resources lazily; when ``render_mode`` is ``None``
        no viewer/renderer is created.  Consequently headless training loops incur
        zero overhead as long as :meth:`render` is never invoked.
        """
        mode = self.render_mode
        if mode is None:
            if self.viewer is not None or self.renderer is not None:
                self._destroy_render_backend()
            return None

        if not (force or self.should_render()):
            return None

        self._ensure_render_backend()

        if mode == "human":
            return self._render_human()
        if mode == "rgb_array":
            return self._render_rgb_array()
        raise RuntimeError(f"Unsupported render_mode: {mode}")

    def _ensure_render_backend(self) -> None:
        """Create viewer/renderer on demand."""
        mode = self.render_mode
        if mode is None:
            return

        if mode == "human":
            if self.viewer is None:
                self.create_viewer()
        elif mode == "rgb_array":
            if self.renderer is None:
                self.create_viewer()
        else:
            raise RuntimeError(f"Unsupported render_mode: {mode}")

    def _destroy_render_backend(self) -> None:
        """Tear down viewer/renderer resources."""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug("Viewer close warning: %s", exc)
            self.viewer = None

        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug("Renderer close warning: %s", exc)
            self.renderer = None

    def _render_human(self):
        if self.viewer is None:
            raise RuntimeError("Human mode requested but viewer is not initialised.")

        target_fps = self.metadata.get("render_fps", None)
        if target_fps:
            interval = 1.0 / float(target_fps)
            now = time.perf_counter()
            last = self._last_human_render
            if last is not None:
                spare = interval - (now - last)
                if spare > 0.0:
                    time.sleep(spare)
            self._last_human_render = now

        self.viewer.sync()
        return None

    def _render_rgb_array(self):
        if self.renderer is None:
            raise RuntimeError("rgb_array mode requested but renderer is not initialised.")

        sim_state = self._get_simulation_state()
        camera = self._get_current_camera()
        self.renderer.update_scene(sim_state, camera=camera)
        frame = self.renderer.render()
        return frame

    # ------------------------------------------------------------------
    # Abstract helpers for subclasses
    # ------------------------------------------------------------------
    def create_viewer(self) -> None:
        """
        Subclasses must create viewer/renderer instances based on :attr:`render_mode`.
        """
        raise NotImplementedError("create_viewer() must be implemented by the subclass if using rendering.")

    def key_callback(self, keycode: int) -> None:
        """Keyboard event handler for interactive viewers (override as needed)."""

    def _get_simulation_state(self):
        """
        For ``rgb_array`` mode: return the simulator state used by the renderer.
        Must be overridden by subclasses that support off-screen rendering.
        """
        raise NotImplementedError("_get_simulation_state() must be implemented by the subclass for rgb_array.")

    def _get_current_camera(self):
        """
        Select the active camera for ``rgb_array`` mode. By default, returns camera index 0.
        """
        return 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment state. Must be implemented by subclasses."""
        raise NotImplementedError("'reset()' must be implemented by the subclass.")

    def step(self, action):
        """Advance the environment dynamics by one step. Must be implemented by subclasses."""
        raise NotImplementedError("step() must be implemented by the subclass.")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        """
        Release viewer/renderer handles and execute registered cleanup callbacks.
        Persistent render options remain untouched so callers can reuse their
        configuration after reopening viewers.  Subclasses overriding this method
        should call ``super().close()`` last.
        """
        while self._resource_stack:
            cleanup = self._resource_stack.pop()
            try:
                cleanup()
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug("Cleanup warning: %s", exc)

        self._destroy_render_backend()
