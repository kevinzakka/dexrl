import dataclasses

import dm_env
from acme import wrappers
from dexterity import manipulation
from dm_control import suite
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


def make_environment(
    domain_name: str,
    task_name: str,
    seed: int,
) -> dm_env.Environment:
    env = manipulation.load(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        # task_kwargs=dict(random=seed),
    )

    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.ConcatObservationWrapper(env)

    return env
