import dataclasses

import dm_env
from acme import wrappers
from dexterity import manipulation
from dm_env import specs


@dataclasses.dataclass(frozen=True)
class EnvironmentSpec:
    observation_spec: specs.Array
    action_spec: specs.Array

    @staticmethod
    def make(env: dm_env.Environment) -> "EnvironmentSpec":
        return EnvironmentSpec(
            observation_spec=env.observation_spec(),
            action_spec=env.action_spec(),
        )


def make_env(domain_name: str, task_name: str, seed: int) -> dm_env.Environment:
    env = manipulation.load(domain_name=domain_name, task_name=task_name, seed=seed)

    env = wrappers.ConcatObservationWrapper(env)  # Convert obs from dict to flat array.
    env = wrappers.CanonicalSpecWrapper(env, clip=True)  # Rescale acts to [-1, 1].
    env = wrappers.SinglePrecisionWrapper(env)  # Single precision obs and acts.

    return env
