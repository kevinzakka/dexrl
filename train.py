import jax
from absl import app
from absl import flags
from absl import logging

import env_utils
import replay_buffer
import td3_lib
import wandb

flags.DEFINE_string("domain_name", "reacher", "Domain name.")
flags.DEFINE_string("task_name", "easy", "Task name.")
flags.DEFINE_integer("seed", 0, "RNG seed.")
flags.DEFINE_integer("num_episodes", 300, "How many episodes to train for.")
flags.DEFINE_integer("start_training", 5, "How many episodes to train for.")
flags.DEFINE_integer("eval_every_n_episodes", 10, "How often to evaluate.")
flags.DEFINE_integer("eval_episodes", 10, "How many eval episodes to run.")

FLAGS = flags.FLAGS


def main(_) -> None:
    wandb.init(entity="kzakka", project="td3")

    # Create environment and grab its specs.
    environment = env_utils.make_environment(
        FLAGS.domain_name, FLAGS.task_name, FLAGS.seed
    )
    eval_environment = env_utils.make_environment(
        FLAGS.domain_name, FLAGS.task_name, FLAGS.seed + 42
    )
    spec = env_utils.EnvironmentSpec.make(environment)

    # TD3 hyperparameters.
    config = td3_lib.TD3Config()

    wandb.config.update(FLAGS)
    wandb.config.update(config)

    # Replay buffer.
    accumulator = replay_buffer.ReplayBuffer(
        capacity=config.replay_size, discount_factor=config.discount
    )

    prng = jax.random.PRNGKey(FLAGS.seed)
    global_step = 0

    # Initialize state.
    prng, init_prng = jax.random.split(prng)
    state = td3_lib.TrainState.initialize(config=config, spec=spec, prng_key=init_prng)

    logging.info(f"Training for {FLAGS.num_episodes} episodes.")
    for episode in range(FLAGS.num_episodes):
        # Reset episode.
        timestep = environment.reset()
        accumulator.insert(timestep, None)

        while not timestep.last():
            # Act.
            prng, action_rng = jax.random.split(prng)
            if episode < FLAGS.start_training:
                action = td3_lib.action_spec_sample(spec.action_spec, action_rng)
            else:
                action = state.policy_step(timestep, action_rng)

            # Agent-environment interaction.
            timestep = environment.step(action)

            # Store experience.
            accumulator.insert(timestep, action)  # type: ignore
            global_step += 1

            if accumulator.is_ready(config.batch_size):
                prng, step_rng = jax.random.split(prng)
                state, train_info = state.learner_step(
                    transitions=accumulator.sample(config.batch_size),
                    prng_key=step_rng,
                )
                for k, v in train_info.items():
                    wandb.log({f"training/{k}": v}, global_step)

        # Evaluate.
        if episode % FLAGS.eval_every_n_episodes == 0:
            eval_info = td3_lib.evaluate(eval_environment, state, FLAGS.eval_episodes)
            for k, v in eval_info.items():
                print(f"{k}: {v}")
                wandb.log({f"evaluation/{k}": v}, global_step)


if __name__ == "__main__":
    app.run(main)
