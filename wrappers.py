import pathlib
from collections import deque
from typing import Deque, List, Optional

import dm_env
import imageio
import numpy as np
from acme.wrappers import base


class TrackEpisodeStatsWrapper(base.EnvironmentWrapper):
    """A wrapper that tracks an episode's return and length."""

    def __init__(self, environment: dm_env.Environment, deque_size: int = 100) -> None:
        super().__init__(environment)

        self._episode_return = 0.0
        self._episode_length = 0
        self._return_queue = deque(maxlen=deque_size)
        self._length_queue = deque(maxlen=deque_size)

    def reset(self) -> dm_env.TimeStep:
        self._episode_return = 0.0
        self._episode_length = 0
        return self._environment.reset()

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        self._episode_return += timestep.reward
        self._episode_length += 1
        if timestep.last():
            self._return_queue.append(self._episode_return)
            self._length_queue.append(self._episode_length)
            self._episode_return = 0.0
            self._episode_length = 0
        return timestep

    @property
    def return_queue(self) -> Deque:
        return self._return_queue

    @property
    def length_queue(self) -> Deque:
        return self._length_queue


class VideoWrapper(base.EnvironmentWrapper):
    """A wrapper for rendering MuJoCo-based environments."""

    def __init__(
        self,
        environment: dm_env.Environment,
        path: str,
        frame_rate: Optional[int] = None,
        camera_id: Optional[int] = 0,
        height: int = 240,
        width: int = 320,
        playback_speed: float = 1.0,
        record_every: int = 100,
    ) -> None:
        if not hasattr(environment, "physics"):
            raise ValueError("Environment expected to have a physics object.")

        super().__init__(environment)

        if frame_rate is None:
            control_timestep = getattr(environment, "control_timestep")()  # noqa: B009
            frame_rate = int(round(playback_speed / control_timestep))

        self._frame_rate = frame_rate
        self._camera_id = camera_id
        self._height = height
        self._width = width
        self._record_every = record_every
        self._path = pathlib.Path(path)

        self._episode_counter = 0
        self._frames: List[np.ndarray] = []

        # Ensure the directory exists.
        if not self._path.exists():
            self._path.mkdir(parents=True)

    def step(self, action) -> dm_env.TimeStep:
        timestep = self.environment.step(action)
        self._append_frame()
        return timestep

    def reset(self) -> dm_env.TimeStep:
        if self._frames:
            self._write_frames()
        self._episode_counter += 1
        timestep = self.environment.reset()
        self._append_frame()
        return timestep

    def close(self):
        if self._frames:
            self._write_frames()
            self._frames = []
        self._environment.close()

    # Helper methods.

    def _render_frame(self) -> np.ndarray:
        if hasattr(self.environment, "frames"):
            return self.environment.frames
        return self.environment.physics.render(
            camera_id=self._camera_id,
            height=self._height,
            width=self._width,
        )

    def _write_frames(self) -> None:
        if self._episode_counter % self._record_every == 0:
            filename = self._path / f"{self._episode_counter:08d}.mp4"
            imageio.mimsave(filename, self._frames, fps=self._frame_rate)
        self._frames = []

    def _append_frame(self) -> None:
        frames = self._render_frame()
        if isinstance(frames, np.ndarray):
            self._frames.append(self._render_frame())
        else:
            self._frames.extend(frames)
