import dm_env
# from acme import wrappers
from dexterity import manipulation


def make_environment(domain_name: str, task_name: str, seed: int) -> dm_env.Environment:
    env = manipulation.load(domain_name=domain_name, task_name=task_name, seed=seed)
    # env = wrappers.SinglePrecisionWrapper(env)  # float32 observations and actions.
    # env = wrappers.CanonicalSpecWrapper(env, clip=True)  # action space is [-1, 1].
    # env = wrappers.ConcatObservationWrapper(env)  # observation dict to array.
    return env
