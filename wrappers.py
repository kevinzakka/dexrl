from collections import deque
from typing import Deque

import dm_env
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
