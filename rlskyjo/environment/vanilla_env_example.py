from rlskyjo.environment import skyjo_env
from rlskyjo.game.skyjo import SkyjoGame
from rlskyjo.models.random_admissible_policy import random_admissible_policy


def simple_episode(config):
    """create a vanilla example"""

    env_pettingzoo = skyjo_env.env(**config)

    env_pettingzoo.reset()

    for agent in env_pettingzoo.agent_iter(max_iter=200 * config["num_players"]):
        # get observation (state) for current agent:
        obs, reward, done, info = env_pettingzoo.last()

        # store current state
        if not done:
            observation = obs["observations"]
            action_mask = obs["action_mask"]
            # action given observation
            action = random_admissible_policy(observation, action_mask)
            print(f"{agent} : {SkyjoGame.render_action_explainer(action_int=action)}")
            # perform action
            env_pettingzoo.step(action)
            # show action
            env_pettingzoo.render()
        else:
            # agent is done -> all agents are done
            env_pettingzoo.step(None)

            # improve policy on reward:
            print(f"{agent} reward: {reward}")
    print("episode done.")


if __name__ == "__main__":
    env_config = {
        "num_players": 2,
        "score_penalty": 2.0,
        "observe_other_player_indirect": True,
        "mean_reward": 1.0,
        "reward_refunded": 0.001,
    }
    simple_episode(config=env_config)
