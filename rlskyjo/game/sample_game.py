from rlskyjo.game.skyjo import SkyjoGame
from rlskyjo.models.random_admissible_policy import policy_ra


def sample_run(games=5000, verbose=0, config={"num_players": 2}):
    skyjo_obj_game = SkyjoGame(**config)
    for _ in range(games):
        rnd = 0
        skyjo_obj_game.reset()
        while not skyjo_obj_game.is_terminated:
            rnd += 1
            player_id, action_expected = skyjo_obj_game.expected_action
            obs, action_mask = skyjo_obj_game.collect_observation(player_id)

            # pick a valid random action
            action = policy_ra(obs, action_mask)
            if verbose:
                print(skyjo_obj_game.render_table())
                print(skyjo_obj_game.render_action_explainer(action))

            won = skyjo_obj_game.act(player_id, action)

            if verbose:
                print(skyjo_obj_game.render_table())
        else:
            # upon termination
            if verbose:
                print(skyjo_obj_game.render_table())


if __name__ == "__main__":
    sample_run()
