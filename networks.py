import dataclasses

import flax.linen as nn
import jax.numpy as jnp

from env_utils import EnvironmentSpec

relu_layer_init = nn.initializers.kaiming_normal()
linear_layer_init = nn.initializers.lecun_normal()
tanh_layer_init = nn.initializers.glorot_normal()


class Actor(nn.Module):
    """A feedforward actor network."""

    action_dim: int

    @nn.compact
    def __call__(  # type: ignore
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        *batch_axes, _ = x.shape

        # Layer 1.
        x = nn.Dense(256, kernel_init=relu_layer_init)(x)
        x = nn.relu(x)
        assert x.shape == (*batch_axes, 256)

        # Layer 2.
        x = nn.Dense(256, kernel_init=relu_layer_init)(x)
        x = nn.relu(x)
        assert x.shape == (*batch_axes, 256)

        # Layer 3.
        x = nn.Dense(self.action_dim, kernel_init=tanh_layer_init)(x)
        action = nn.tanh(x)  # B, A.
        assert action.shape == (*batch_axes, self.action_dim)

        return action


class Critic(nn.Module):
    """A feedforward critic network."""

    @nn.compact
    def __call__(  # type: ignore
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        *batch_axes, _ = states.shape

        x = jnp.concatenate([states, actions], axis=-1)
        assert x.shape == (*batch_axes, states.shape[-1] + actions.shape[-1])

        # Layer 1.
        x = nn.Dense(256, use_bias=False, kernel_init=tanh_layer_init)(x)
        x = nn.LayerNorm(use_scale=True, use_bias=True)(x)
        x = nn.tanh(x)
        assert x.shape == (*batch_axes, 256)

        # Layer 2.
        x = nn.Dense(256, kernel_init=relu_layer_init)(x)
        x = nn.relu(x)
        assert x.shape == (*batch_axes, 256)

        # Layer 3.
        x = nn.Dense(1, kernel_init=linear_layer_init)(x)
        q_value = jnp.squeeze(x, axis=-1)
        assert q_value.shape == (*batch_axes,)

        return q_value


@dataclasses.dataclass
class TD3Networks:
    policy: Actor
    critic: Critic
    twin_critic: Critic

    @staticmethod
    def initialize(spec: EnvironmentSpec) -> "TD3Networks":
        return TD3Networks(
            policy=Actor(action_dim=spec.action.shape[0]),
            critic=Critic(),
            twin_critic=Critic(),
        )
