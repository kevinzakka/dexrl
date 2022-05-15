import flax.linen as nn
import jax
import jax.numpy as jnp


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
        return x  # B, 1


class DoubleCritic(nn.Module):
    @nn.compact
    def __call__(self, states, actions):
        critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        return critic()(states, actions)  # 2, B, 1.
