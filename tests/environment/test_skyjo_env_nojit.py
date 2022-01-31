from numba import config

config.DISABLE_JIT = True
from itertools import product

import pytest

from rlskyjo.environment.vanilla_env_example import simple_episode


@pytest.mark.last
def test_skyjoev():

    
    
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
    
