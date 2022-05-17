import dataclasses

import dm_env
import jax
import jax.numpy as jnp
from acme import wrappers
from dexterity import manipulation
from dm_env import specs


@dataclasses.dataclass(frozen=True)
class EnvironmentSpec:
    observation: specs.Array
    action: specs.Array

    @staticmethod
    def make(env: dm_env.Environment) -> "EnvironmentSpec":
        return EnvironmentSpec(
            observation=env.observation_spec(),
            action=env.action_spec(),
        )

    @staticmethod
    def zeros_like(spec: specs.Array) -> jnp.ndarray:
        return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), spec)


def make_env(domain_name: str, task_name: str, seed: int) -> dm_env.Environment:
    env = manipulation.load(domain_name, task_name, seed=seed)

    env = wrappers.ConcatObservationWrapper(env)  # Convert obs from dict to flat array.
    env = wrappers.CanonicalSpecWrapper(env, clip=True)  # Rescale acts to [-1, 1].
    env = wrappers.SinglePrecisionWrapper(env)  # Single precision obs and acts.

    return env
