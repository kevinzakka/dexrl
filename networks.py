import dataclasses

import flax.linen as nn
import jax.numpy as jnp

from env_utils import EnvironmentSpec


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return nn.tanh(x)  # B, A.


class Critic(nn.Module):
    @nn.compact
    def __call__(self, states, actions):
        x = jnp.concatenate([states, actions], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x)  # (B,).


@dataclasses.dataclass
class TD3Networks:
    policy: Actor
    critic: Critic
    twin_critic: Critic

    @staticmethod
    def initialize(spec: EnvironmentSpec) -> "TD3Networks":
        return TD3Networks(
            policy=Actor(action_dim=spec.action_spec.shape[0]),
            critic=Critic(),
            twin_critic=Critic(),
        )
