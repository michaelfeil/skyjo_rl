from numba import config

config.DISABLE_JIT = True


from rlskyjo.environment.skyjo_env import DEFAULT_CONFIG, env
from rlskyjo.environment.vanilla_env_example import simple_episode


def test_skyjo_env_options():

    from itertools import product

    def build_config(
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
        config = build_config(*options)
        print(count, config)
        simple_episode(config, verbose=(1 if count % 50 == 0 else 0))


if __name__ == "__main__":
    test_skyjo_env_options()
