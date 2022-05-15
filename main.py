import jax
import numpy as np

import agent
import helpers
import replay_buffer

SEED = 0
TRAIN_EPISODES = 0  # How many steps to train for.
START_TRAIN_AT = 10  # When to start training.
REPLAY_SIZE = 1_000_000
BATCH_SIZE = 256
DISCOUNT_FACTOR = 0.99


def main() -> None:
    environment = helpers.make_environment("reach", "state_dense", seed=SEED)
    action_spec = environment.action_spec()
    observation_spec = environment.observation_spec()

    rng = jax.random.PRNGKey(SEED)
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)

    buffer = replay_buffer.ReplayBuffer(
        capacity=REPLAY_SIZE, discount_factor=DISCOUNT_FACTOR
    )

    policy = agent.Agent(
        observation_spec=observation_spec,
        action_spec=action_spec,
        actor_rng=actor_rng,
        critic_rng=critic_rng,
    )

    for episode in range(TRAIN_EPISODES):

        timestep = environment.reset()
        buffer.insert(timestep, None)

        while not timestep.last():
            if episode < START_TRAIN_AT:
                action = np.random.uniform(
                    low=action_spec.minimum,
                    high=action_spec.maximum,
                    size=action_spec.shape,
                ).astype(action_spec.dtype)
            else:
                action = None

            timestep = environment.step(action)
            buffer.insert(timestep, action)

            if buffer.is_ready(BATCH_SIZE) and episode >= START_TRAIN_AT:
                # Update the agent.
                pass

        # Evaluate.


if __name__ == "__main__":
    main()
