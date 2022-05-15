import collections
import random
from typing import Deque, NamedTuple, Optional

import dm_env
import numpy as np
from dm_env import specs


class Transition(NamedTuple):
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    next_observation: np.ndarray


class ReplayBuffer:
    """A dead-simple in-memory replay buffer."""

    def __init__(self, capacity: int, discount_factor: float) -> None:
        self._prev: Optional[dm_env.TimeStep] = None
        self._action: Optional[np.ndarray] = None
        self._latest: Optional[dm_env.TimeStep] = None
        self._buffer: Deque = collections.deque(maxlen=capacity)
        self._discount_factor = discount_factor

    def __len__(self) -> int:
        return len(self._buffer)

    def insert(self, timestep: dm_env.TimeStep, action: Optional[np.ndarray]) -> None:
        self._prev = self._latest
        self._action = action
        self._latest = timestep

        if action is not None:
            self._buffer.append(
                (
                    self._prev.observation,  # type: ignore
                    self._action,
                    self._latest.reward,
                    self._latest.discount,
                    self._latest.observation,
                )
            )

    def sample(self, batch_size: int) -> Transition:
        obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
            *random.sample(self._buffer, batch_size)
        )

        return Transition(
            np.stack(obs_tm1),
            np.asarray(a_tm1),
            np.asarray(r_t),
            np.asarray(discount_t) * self._discount_factor,
            np.stack(obs_t),
        )

    def is_ready(self, batch_size: int) -> bool:
        return batch_size <= len(self._buffer)
