import jax
from dm_env import specs
import jax.numpy as jnp

import networks


class Agent:
    def __init__(
        self,
        observation_spec: specs.Array,
        action_spec: specs.Array,
        actor_rng: jax.random.KeyArray,
        critic_rng: jax.random.KeyArray,
    ) -> None:

        # Initialize actor.
        self._actor = networks.Actor(action_dim=action_spec.shape[0])
        obs_sample = jnp.ones((1, observation_spec.shape[0]))
        self._actor_params = self._actor.init(actor_rng, obs_sample)

        # Initialize critics.
        self._critic = networks.DoubleCritic()
        action_sample = jnp.ones((1, action_spec.shape[0]))
        self._critic_params = self._critic.init(critic_rng, obs_sample, action_sample)

