import random
import time

import jax
import tqdm
from absl import app, flags

import env_utils
import evaluation
import replay_buffer
import td3
import wrappers

flags.DEFINE_string("experiment_name", None, "Name for this experiment.")
flags.DEFINE_string("domain_name", "reach", "Domain name.")
flags.DEFINE_string("task_name", "state_dense", "Task name.")
flags.DEFINE_bool("use_wandb", False, "Use wandb for logging.")
flags.DEFINE_integer("seed", 0, "RNG seed.")
flags.DEFINE_integer("total_timesteps", int(1e6), "Timesteps to train for.")
flags.DEFINE_integer("warmstart_timesteps", int(1e4), "Random action timesteps.")
flags.DEFINE_integer("eval_episodes", 10, "How many eval episodes to run.")
flags.DEFINE_integer("eval_interval", int(5e3), "Policy evaluation interval.")
flags.DEFINE_integer("log_interval", int(1e3), "Logging interval.")
flags.DEFINE_integer("checkpoint_interval", int(1e5), "Checkpoint interval.")
flags.DEFINE_bool(
    "capture_video", False, "Capture and save videos during policy evaluation."
)

FLAGS = flags.FLAGS


def string_from_kwargs(**kwargs) -> str:
    return "__".join([f"{k}={v}" for k, v in kwargs.items()])


def main(_) -> None:
    # Generate a unique experiment name.
    run_name = string_from_kwargs(
        exp_name=FLAGS.experiment_name,
        domain_name=FLAGS.domain_name,
        task_name=FLAGS.task_name,
        seed=FLAGS.seed,
        time=int(time.time()),
    )

    # Seed RNGs.
    random.seed(FLAGS.seed)
    rng = jax.random.PRNGKey(FLAGS.seed)

    # TD3 hyperparameters.
    config = td3.TD3Config()

    if FLAGS.use_wandb:
        import wandb

        wandb.init(entity="kzakka", project="td3", name=run_name)
        wandb.config.update(FLAGS)
        wandb.config.update(config)

    # Make train and eval environments.
    environment = env_utils.make_env(FLAGS.domain_name, FLAGS.task_name, FLAGS.seed)
    environment = wrappers.TrackEpisodeStatsWrapper(environment, deque_size=1)
    eval_environment = env_utils.make_env(
        FLAGS.domain_name, FLAGS.task_name, FLAGS.seed + 42
    )
    if FLAGS.capture_video:
        eval_environment = wrappers.VideoWrapper(
            eval_environment,
            path=f"videos/{run_name}",
            record_every=FLAGS.eval_episodes,
        )

    # Replay buffer initialization.
    accumulator = replay_buffer.ReplayBuffer(
        capacity=config.replay_size, discount_factor=config.discount
    )

    # TD3 agent initialization.
    rng, init_prng = jax.random.split(rng)
    spec = env_utils.EnvironmentSpec.make(environment)
    state = td3.TrainState.initialize(config=config, spec=spec, rng_key=init_prng)

    try:
        timestep = environment.reset()
        accumulator.insert(timestep, None)

        for i in tqdm.tqdm(range(FLAGS.total_timesteps), smoothing=0.1):
            # Act.
            rng, action_rng = jax.random.split(rng)
            if i < FLAGS.warmstart_timesteps:
                action = td3.action_spec_sample(spec.action_spec, action_rng)
            else:
                action = state.policy_step(timestep, action_rng)

            # Interact.
            timestep = environment.step(action)

            # Store experience.
            accumulator.insert(timestep, action)  # type: ignore

            # Reset
            if timestep.last():
                if FLAGS.use_wandb:
                    wandb.log({"training/return": environment.return_queue[0]}, i)
                    wandb.log({"training/length": environment.length_queue[0]}, i)

                timestep = environment.reset()
                accumulator.insert(timestep, None)

            # Update.
            if accumulator.is_ready(config.batch_size):
                rng, step_rng = jax.random.split(rng)
                state, train_info = state.learner_step(
                    transitions=accumulator.sample(config.batch_size),
                    rng_key=step_rng,
                )

                if i % FLAGS.log_interval == 0:
                    if FLAGS.use_wandb:
                        for k, v in train_info.items():
                            wandb.log({f"training/{k}": v}, i)

            # Evaluate.
            # Note: This will evaluate at the very first step.
            if i % FLAGS.eval_interval == 0:
                eval_info = evaluation.evaluate(
                    eval_environment, state, FLAGS.eval_episodes
                )
                if FLAGS.use_wandb:
                    wandb.log({"evaluation/return": eval_info.average_return}, i)
                    wandb.log({"evaluation/length": eval_info.average_length}, i)

            # Checkpoint.
            if i % FLAGS.checkpoint_interval == 0:
                pass

    except KeyboardInterrupt:
        print("Caught keyboard interrupt. Saving before quitting.")

    finally:
        # Save checkpoint.
        pass


if __name__ == "__main__":
    flags.mark_flag_as_required("experiment_name")
    app.run(main)
