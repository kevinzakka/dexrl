from typing import NamedTuple

import dm_env
import numpy as np

from td3 import TrainState
from wrappers import TrackEpisodeStatsWrapper


class EvaluationMetrics(NamedTuple):
    average_return: float
    average_length: float


def evaluate(
    environment: dm_env.Environment,
    state: TrainState,
    num_episodes: int,
) -> EvaluationMetrics:
    """Evaluates the environment by rolling out the policy for a number of episodes."""

    environment = TrackEpisodeStatsWrapper(environment)

    for _ in range(num_episodes):
        timestep = environment.reset()
        while not timestep.last():
            action = state.get_action(timestep)
            timestep = environment.step(action)

    return EvaluationMetrics(
        average_return=np.mean(environment.return_queue),
        average_length=np.mean(environment.length_queue),
    )
