import dataclasses
from typing import Dict, Tuple

import dm_env
import flax
import jax
import jax.numpy as jnp
import optax
import rlax
from dm_env import specs
from flax import struct

from env_utils import EnvironmentSpec
from networks import TD3Networks
from replay_buffer import Transition

# Aliases.
Params = flax.core.frozen_dict.FrozenDict


@dataclasses.dataclass(frozen=True)
class TD3Config:
    # Loss options.
    batch_size: int = 256
    policy_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    discount: float = 0.99

    # TD3 specific options.
    sigma: float = 0.1
    delay: int = 2
    target_sigma: float = 0.2
    noise_clip: float = 0.5
    tau: float = 0.005

    # Replay options.
    replay_size: int = int(1e6)


@struct.dataclass
class TrainState:
    """TD3 trainer."""

    policy_params: Params
    target_policy_params: Params
    critic_params: Params
    target_critic_params: Params
    twin_critic_params: Params
    target_twin_critic_params: Params
    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    twin_critic_opt_state: optax.OptState
    steps: int

    config: TD3Config = struct.field(pytree_node=False)
    spec: EnvironmentSpec = struct.field(pytree_node=False)
    networks: TD3Networks = struct.field(pytree_node=False)
    policy_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    twin_critic_optimizer: optax.GradientTransformation = struct.field(
        pytree_node=False
    )

    @staticmethod
    def initialize(
        config: TD3Config,
        spec: EnvironmentSpec,
        rng_key: jax.random.KeyArray,
    ) -> "TrainState":
        critic_rng, twin_critic_rng, policy_rng = jax.random.split(rng_key, 3)

        sample_obs = jnp.expand_dims(spec.zeros_like(spec.observation), 0)
        sample_act = jnp.expand_dims(spec.zeros_like(spec.action), 0)

        networks = TD3Networks.initialize(spec)

        # Critic.
        initial_critic_params = networks.critic.init(critic_rng, sample_obs, sample_act)
        initial_target_critic_params = initial_critic_params

        # Twin critic.
        initial_twin_critic_params = networks.twin_critic.init(
            twin_critic_rng, sample_obs, sample_act
        )
        initial_target_twin_critic_params = initial_twin_critic_params

        # Policy.
        initial_policy_params = networks.policy.init(policy_rng, sample_obs)
        initial_target_policy_params = initial_policy_params

        # Optimizers.
        policy_optimizer = optax.adam(config.policy_learning_rate)
        initial_policy_opt_state = policy_optimizer.init(initial_policy_params)
        critic_optimizer = optax.adam(config.critic_learning_rate)
        initial_critic_opt_state = critic_optimizer.init(initial_critic_params)
        twin_critic_optimizer = optax.adam(config.critic_learning_rate)
        initial_twin_critic_opt_state = twin_critic_optimizer.init(
            initial_twin_critic_params
        )

        return TrainState(
            config=config,
            spec=spec,
            steps=0,
            # Networks and parameters.
            networks=networks,
            policy_params=initial_policy_params,
            target_policy_params=initial_target_policy_params,
            critic_params=initial_critic_params,
            target_critic_params=initial_target_critic_params,
            twin_critic_params=initial_twin_critic_params,
            target_twin_critic_params=initial_target_twin_critic_params,
            # Optimizers and states.
            policy_optimizer=policy_optimizer,
            policy_opt_state=initial_policy_opt_state,
            critic_optimizer=critic_optimizer,
            critic_opt_state=initial_critic_opt_state,
            twin_critic_optimizer=twin_critic_optimizer,
            twin_critic_opt_state=initial_twin_critic_opt_state,
        )

    @jax.jit
    def get_action(self, env_output: dm_env.TimeStep) -> jnp.ndarray:
        """Policy step during evaluation."""
        obs = jnp.expand_dims(env_output.observation, 0)  # (1, D).
        # The critic takes care of the squeezing so no need to remove the batch
        # dimension we added above.
        return self.networks.policy.apply(self.policy_params, obs)[0]

    @jax.jit
    def policy_step(
        self,
        env_output: dm_env.TimeStep,
        rng_key: jax.random.KeyArray,
    ) -> jnp.ndarray:
        """Policy step during training."""
        # Forward observation through policy network to get an action.
        action = self.get_action(env_output)

        # Add noise to the action and clip to make sure it remains within bounds.
        return add_policy_noise(
            action,
            self.spec.action,
            rng_key,
            self.config.target_sigma,
            self.config.noise_clip,
        )

    @jax.jit
    def learner_step(
        self,
        transitions: Transition,
        rng_key: jax.random.KeyArray,
    ) -> Tuple["TrainState", Dict[str, jnp.ndarray]]:
        key_critic, key_twin = jax.random.split(rng_key, 2)

        def polyak_average(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """Exponential average of network and target parameters."""
            return x * (1 - self.config.tau) + y * self.config.tau

        def critic_loss(
            critic_params: Params, rng_key: jax.random.KeyArray
        ) -> jnp.ndarray:
            q_tm1 = self.networks.critic.apply(
                critic_params, transitions.observation, transitions.action
            )
            action = self.networks.policy.apply(
                self.target_policy_params, transitions.next_observation
            )
            action = add_policy_noise(
                action,
                self.spec.action,
                rng_key,
                self.config.target_sigma,
                self.config.noise_clip,
            )
            q_t = self.networks.critic.apply(
                self.target_critic_params, transitions.next_observation, action
            )
            twin_q_t = self.networks.twin_critic.apply(
                self.target_twin_critic_params, transitions.next_observation, action
            )
            q_t = jnp.minimum(q_t, twin_q_t)
            target_q_tm1 = (
                transitions.reward + self.config.discount * transitions.discount * q_t
            )
            td_error = jax.lax.stop_gradient(target_q_tm1) - q_tm1
            return jnp.mean(jnp.square(td_error))

        def policy_loss(policy_params: Params, critic_params: Params) -> jnp.ndarray:
            """Calculates the deterministic policy gradient (DPG) loss."""
            action = self.networks.policy.apply(policy_params, transitions.observation)
            grad_critic = jax.vmap(
                jax.grad(self.networks.critic.apply, argnums=2),
                in_axes=(None, 0, 0),
            )
            dq_da = grad_critic(critic_params, transitions.observation, action)
            batch_dpg_learning = jax.vmap(rlax.dpg_loss, in_axes=(0, 0))
            loss = batch_dpg_learning(action, dq_da)
            return jnp.mean(loss)

        critic_loss_and_grad = jax.value_and_grad(critic_loss)

        # Critic update.
        critic_loss_value, critic_gradients = critic_loss_and_grad(
            self.critic_params, key_critic
        )
        critic_updates, critic_opt_state = self.critic_optimizer.update(
            critic_gradients, self.critic_opt_state
        )
        critic_params = optax.apply_updates(self.critic_params, critic_updates)
        target_critic_params = jax.tree_util.tree_map(
            polyak_average, self.target_critic_params, critic_params
        )

        # Twin critic update.
        twin_critic_loss_value, twin_critic_gradients = critic_loss_and_grad(
            self.twin_critic_params, key_twin
        )
        twin_critic_updates, twin_critic_opt_state = self.twin_critic_optimizer.update(
            twin_critic_gradients, self.twin_critic_opt_state
        )
        twin_critic_params = optax.apply_updates(
            self.twin_critic_params, twin_critic_updates
        )
        target_twin_critic_params = jax.tree_util.tree_map(
            polyak_average, self.target_twin_critic_params, twin_critic_params
        )

        # Policy update.
        policy_loss_and_grad = jax.value_and_grad(policy_loss)
        policy_loss_value, policy_gradients = policy_loss_and_grad(
            self.policy_params, self.critic_params
        )

        def update_policy_step() -> Tuple[Params, Params, optax.OptState]:
            policy_updates, policy_opt_state = self.policy_optimizer.update(
                policy_gradients, self.policy_opt_state
            )
            policy_params = optax.apply_updates(self.policy_params, policy_updates)
            target_policy_params = jax.tree_util.tree_map(
                polyak_average, self.target_policy_params, policy_params
            )
            return policy_params, target_policy_params, policy_opt_state

        # Only applied every `delay` steps.
        current_policy_state = (
            self.policy_params,
            self.target_policy_params,
            self.policy_opt_state,
        )
        policy_params, target_policy_params, policy_opt_state = jax.lax.cond(
            self.steps % self.config.delay == 0,
            lambda _: update_policy_step(),
            lambda _: current_policy_state,
            operand=None,
        )

        new_state = TrainState(
            config=self.config,
            spec=self.spec,
            networks=self.networks,
            policy_params=policy_params,
            target_policy_params=target_policy_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            twin_critic_params=twin_critic_params,
            target_twin_critic_params=target_twin_critic_params,
            policy_optimizer=self.policy_optimizer,
            policy_opt_state=policy_opt_state,
            critic_optimizer=self.critic_optimizer,
            critic_opt_state=critic_opt_state,
            twin_critic_optimizer=self.twin_critic_optimizer,
            twin_critic_opt_state=twin_critic_opt_state,
            steps=self.steps + 1,
        )

        metrics = {
            "policy_loss": policy_loss_value,
            "critic_loss": critic_loss_value,
            "twin_critic_loss": twin_critic_loss_value,
        }

        return new_state, metrics

    def save_checkpoint(self, filename: str) -> None:
        """Saves the current model state to a checkpoint."""
        raise NotImplementedError

    def load_checkpoint(self, filename: str) -> "TrainState":
        """Loads the model state from a checkpoint."""
        raise NotImplementedError


def add_policy_noise(
    action: jnp.ndarray,
    action_spec: specs.Array,
    rng_key: jax.random.KeyArray,
    target_sigma: float,
    noise_clip: float,
) -> jnp.ndarray:
    noise = jax.random.normal(key=rng_key, shape=action_spec.shape) * target_sigma
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    return jnp.clip(action + noise, action_spec.minimum, action_spec.maximum)
