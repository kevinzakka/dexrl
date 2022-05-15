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


if __name__ == "__main__":
    model = Actor(action_dim=2)
    obs = jnp.ones((32, 64))
    variables = model.init(jax.random.PRNGKey(0), obs)
    output = model.apply(variables, obs)
    assert output.shape == (32, 2)

    model = Critic()
    obs = jnp.ones((32, 64))
    act = jnp.ones((32, 24))
    variables = model.init(jax.random.PRNGKey(0), obs, act)
    output = model.apply(variables, obs, act)
    assert output.shape == (32, 1)

    model = DoubleCritic()
    obs = jnp.ones((32, 64))
    act = jnp.ones((32, 24))
    variables = model.init(jax.random.PRNGKey(0), obs, act)
    output = model.apply(variables, obs, act)
    assert output.shape == (2, 32, 1)
