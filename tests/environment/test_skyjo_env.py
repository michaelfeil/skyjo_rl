from numba import config

import pytest
from rlskyjo.environment import skyjo_env
from rlskyjo.models.random_admissible_policy import policy_ra
from rlskyjo.environment.skyjo_env import DEFAULT_CONFIG, env
from rlskyjo.environment.vanilla_env_example import simple_episode
import numpy as np

def test_skyjo_env_options():
    config.DISABLE_JIT = True
    from itertools import product

    def build_config_env(
        num_players,
        score_penalty,
        observe_other_player_indirect,
        mean_reward,
        reward_refunded,
    ):
        return {
            "num_players": num_players,
            "score_penalty": score_penalty,
            "observe_other_player_indirect": observe_other_player_indirect,
            "mean_reward": mean_reward,
            "reward_refunded": reward_refunded,
        }

    num_players = list(range(1, 13))
    score_penalty = [1.0, 2.0]
    observe_other_player_indirect = [True, False]
    mean_reward = [-1, 0.0, 1.0]
    reward_refunded = [0.0, 0.01]

    for count, options in enumerate(
        product(
            num_players,
            score_penalty,
            observe_other_player_indirect,
            mean_reward,
            reward_refunded,
        )
    ):
        config_env = build_config_env(*options)
        print(count, config_env)
        simple_episode(config_env, verbose=(1 if count % 50 == 0 else 0))

@pytest.mark.skip
def test_reproducability(seed = 42, n_runs=2):
    """create a vanilla example"""
    config.DISABLE_JIT = False
    rewards = {
        i: [] for i in range(n_runs)
    }
    observations = {
        i: [] for i in range(n_runs)
    }
    config_env = DEFAULT_CONFIG
    for i in range(2):
        env_pettingzoo = skyjo_env.env(**config_env)
        env_pettingzoo.seed(42)
        rng = np.random.default_rng(seed)
        env_pettingzoo.reset()

        for agent in env_pettingzoo.agent_iter(max_iter=300 * config_env["num_players"]):
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
    print(rewards, observations)



if __name__ == "__main__":
    test_reproducability()

