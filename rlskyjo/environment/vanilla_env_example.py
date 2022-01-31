from rlskyjo.environment import skyjo_env
from rlskyjo.game.skyjo import SkyjoGame
from rlskyjo.models.random_admissible_policy import policy_ra


def simple_episode(config, verbose=0):
    """create a vanilla example"""

    env_pettingzoo = skyjo_env.env(**config)

    env_pettingzoo.reset()

    for agent in env_pettingzoo.agent_iter(max_iter=300 * config["num_players"]):
        # get observation (state) for current agent:
        obs, reward, done, info = env_pettingzoo.last()

        # store current state
        if not done:
            observation = obs["observations"]
            action_mask = obs["action_mask"]
            # action given observation
            action = policy_ra(observation, action_mask)
            if verbose:
                print(
                    f"{agent} : {SkyjoGame.render_action_explainer(action_int=action)}"
                )
            # perform action
            env_pettingzoo.step(action)
            # show action
            if verbose:
                env_pettingzoo.render()
        else:
            # agent is done -> all agents are done
            env_pettingzoo.step(None)

            # improve policy on reward:
            if verbose:
                print(f"{agent} reward: {reward}")
    if verbose:
        print("episode done.")


if __name__ == "__main__":
    print(f"running with config: {skyjo_env.DEFAULT_CONFIG}")
    simple_episode(config=skyjo_env.DEFAULT_CONFIG)
