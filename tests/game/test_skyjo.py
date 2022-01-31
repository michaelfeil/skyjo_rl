import timeit

from rlskyjo.game.sample_game import sample_run
from rlskyjo.game.skyjo import SkyjoGame

# class SkyjoGameNoJIT(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         # Start it once for the entire test suite/module
#         numba.config.DISABLE_JIT =  False

#     def test_sample_game():
#         sample_run(games=1, verbose=1, config={"num_players": 3})

#     @classmethod
#     def tearDownClass(cls):
#         numba.config.DISABLE_JIT =  False


def test_sample_game():
    sample_run(games=1, verbose=1, config={"num_players": 3})
    sample_run(games=500)


def test_render():
    game = SkyjoGame()
    game.render_actions()
    game.render_table()
    game.render_player(0)
    game.render_player(player_id=0, render_cards_open=True)
    [game.render_action_explainer(action_int) for action_int in range(0, 26)]


def test_timing():
    sample_run(games=1, verbose=0, config={"num_players": 3})
    time_2500_games = timeit.timeit(
        'sample_run(games=2500, verbose=0, config={"num_players": 3})',
        number=1,
        globals={"sample_run": sample_run},
    )
    print(time_2500_games)
    assert time_2500_games < 180, (
        f"took longer then 180 seconds ({time_2500_games})"
        " to process 2500 games. common is around 20s"
    )
