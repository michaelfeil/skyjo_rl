import numpy as np

from rlskyjo.environment import skyjo_env
from rlskyjo.environment.skyjo_env import DEFAULT_CONFIG
from rlskyjo.models.random_admissible_policy import policy_ra


def test_reproducability(seed=42):
    """create a vanilla example"""
    n_runs=2

    rewards = {i: [] for i in range(n_runs)}
    observations = {i: [] for i in range(n_runs)}
    config_env = DEFAULT_CONFIG
    for i in range(2):
        env_pettingzoo = skyjo_env.env(**config_env)
        env_pettingzoo.seed(42)
        rng = np.random.default_rng(seed)
        env_pettingzoo.reset()

        for agent in env_pettingzoo.agent_iter(
            max_iter=300 * config_env["num_players"]
        ):
            # get observation (state) for current agent:
            obs, reward, done, info = env_pettingzoo.last()

            # store current state
            if not done:
                observation = obs["observations"]
                observations[i].append(observation)
                action_mask = obs["action_mask"]
                # action given observation
                action = policy_ra(observation, action_mask, rng=rng)

                # perform action
                env_pettingzoo.step(action)
            else:
                # agent is done -> all agents are done
                env_pettingzoo.step(None)
                rewards[i].append(reward)
                # improve policy on reward:
    np.testing.assert_array_equal(observations[0], observations[1])
    np.testing.assert_array_equal(rewards[0], rewards[1])


if __name__ == "__main__":
    test_reproducability()