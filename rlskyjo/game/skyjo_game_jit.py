from pickle import TRUE
import numpy as np
import random 

import numba as nb
from numba.experimental import jitclass
from numba import types, typed
from typing import Tuple
import os

DTYPE = np.int64
# os.environ["NUMBA_DEBUG"] = "1"
instance = np.full((3,12), 0, dtype=DTYPE)
# print(nb.typeof(instance))
spec = [
    ('n_players', types.int32),    
    ('fill_masked_value', types.int32),   
    ('game_over', types.boolean),
    ('discard_pile', types.List(types.int_)),
    ('players_masked', nb.int64[:,:]),
    ('players_cards', nb.int64[:,:]),
    ('drawpile', types.List(types.int_)),
]
@jitclass(spec)
class SkyjoGame(object):
    def __init__(self, n_players: int = 2) -> None:
        """"""
        assert n_players < 12, "can be played up to 6/theoretical 12 players"
        # init objects
        self.n_players = n_players
        self.fill_masked_value = 99
        self.game_over = False

    
    def reset(self):
        self.game_over = False
        
        # 150 cards from -2 to 12
        drawpile = np.repeat(np.arange(-2,13, dtype=DTYPE), 10)
        np.random.shuffle(drawpile)

        self.players_masked = np.full((self.n_players,12), 0, dtype=DTYPE)
        self.players_cards = drawpile[:12*self.n_players].reshape(self.n_players,-1)
        self.drawpile = list(drawpile[self.n_players*12:])
        
        # discard_pile: first in last out
        self.discard_pile = [self.drawpile.pop()]
        
        for p in range(self.n_players):
            picked = np.random.choice(12, 2, replace=False)
            self.players_masked[p][picked] = 1
        return None

    def collect_observation_draw(self, player_id: int):
        counts, min_masked, discard_pi = self.observe_global_game_stats()
        return {
            "player": self.observe_player(player_id),
            "pile_counts": counts,
            "min_n_masked": min_masked,
            "top_discard": discard_pi,
        }
        
    def observe_player(self, player_id: int):
        """observe visible cards from player id"""
        # fill dummy value
        cards = np.full_like(
            self.players_cards[player_id], self.fill_masked_value
        )
        # replace thos cards known with the actual value
        cards[self.players_masked[player_id] != 0] = \
            self.players_cards[player_id][self.players_masked[player_id] != 0 ]
            
        return cards
    
    def observe_global_game_stats(self):
        """observe game statistics, features to percieve global game"""
        # map all ever visible cards
        occured_cards = list(range(-2,13))
        for pl in range(self.n_players):
            occured_cards.extend(list(self.players_cards[pl][self.players_masked[pl] == 1]))
        occured_cards.extend(self.discard_pile)
        


        # # get sums of cards for each player
        # sum_of_player_values = [pl_cards.sum() for pl_cards in known_player_cards]
        # # find out what and who has minimum cards
        # min_sum = np.min(sum_of_player_values)
        # min_sum_player_id = np.where(
        #     sum_of_player_values == min_sum
        # )[0]
        
        # find out who has most cards revealed
        n_masked_player_cards = np.sum(self.players_masked == 0, axis =1 )
        min_masked = np.min(n_masked_player_cards)
        # min_masked_player_id = np.where(
        #     n_masked_player_cards == min_masked
        # )[0]
        # # minimum masked is also minimum value
        # min_masked_is_min_sum = any(
        #     bool(p_value in min_masked_player_id) for p_value in min_sum_player_id
        # )
        # count all ever seen cards
        occured_cards = np.array(occured_cards)
        # add range(-2,13) to visualize counts of 0 times occuring cards too
        counts = np.bincount(occured_cards - np.min(occured_cards))
        counts = counts - 1 # fix the (range)counts to actual value
        return counts, min_masked, self.discard_pile[-1]
    
    def draw_card(self, from_drawpile: bool):
        """draw one card from the"""
        assert len(self.discard_pile) + len(self.drawpile) == 150 - 12*self.n_players
        if from_drawpile:
            if not len(self.drawpile):
                self.reshuffle_discard_pile()
            return self.drawpile.pop()
        else:
            return self.discard_pile.pop()
        
    def reshuffle_discard_pile(self):
        """reshuffle discard pile into drawpile"""
        drawpile = np.array(self.discard_pile.copy(), dtype=DTYPE)
        np.random.shuffle(drawpile)
        self.drawpile = list(drawpile)
        self.discard_pile = [
            self.drawpile.pop()
        ]
    
    def play_player(self, player_id: int, picked_card: int, place_to_pos: int) -> Tuple[bool, np.ndarray]:   
        """
        place_to_pos: int between 0 and 11, 0-11 pos, 
        
        returns:
            terminiate_game: bool, True if terminates
            winner_id
        """
        # perform goal check
        game_done = self.player_goal_check(player_id)
        if game_done:
            eval = self.evaluate_game(player_id)
            return True, eval
        
        # replace with one of the 0-11 cards of player
        # unmask new card
        # discard replaced card
        self.discard_pile.append(
            self.players_cards[player_id][place_to_pos]
        )
        self.players_masked[player_id][place_to_pos] = 1
        self.players_cards[player_id][place_to_pos] = picked_card
        return False, np.array([0],dtype=DTYPE)
            
    def player_goal_check(self, player_id):        
        return np.all((self.players_masked[player_id] != 0))
    
    def evaluate_game(self, player_won_id):       
        # reshape to size of game
        score = None
        # shape = list(self.players_cards.shape)
        # cards = self.players_cards.copy().reshape(shape[0],shape[1]//3,3)
        # figure out where we have columns with all same values
        
        # for pl in range(self.n_players):
        #     # numba implementation of
        #     # cards = self.players_cards.copy().reshape(shape[0],shape[1]//3,3)
        #     # cancelout_cols = np.min(cards, axis=2) == np.max(cards, axis=2)
        #     # cards[~cancelout_cols, :] = 0
        #     cancelout_cols = None
        #     cards_pl = self.players_cards[pl].copy()
        #     for i in range(cards_pl.shape[0]//3):
        #         curr_row = cards_pl[i*3:3+i*3]
        #         cancelout = [int(np.min(curr_row) != np.max(curr_row))]*3
        #         if cancelout_cols is None: 
        #             cancelout_cols = cancelout
        #         else:
        #             cancelout_cols.extend(cancelout)

        #     countable = cards_pl[cancelout_cols]
        #     append = np.sum(countable)
        #     if score is None: 
        #         score = [append]
        #     else:
        #         score.append(append)
            
        # # penalty if finisher is not winner.
        # if np.min(score) != score[player_won_id]:
        #     score[player_won_id] *= 2
        return score
    
def test_game(game):
    
    from_pile = True
    rnd = 0
    
    game.reset()
    won = False
    while not won:
        rnd += 1
        for player in range(game.n_players):
            from_pile = not from_pile 
            position = random.randint(0,11)
            globals_ = game.observe_global_game_stats()
            drawn_card = game.draw_card(from_pile)
            # print("drawn_card", drawn_card)
            before_cards = game.observe_player(player)
            won, stats = game.play_player(player, drawn_card, position)
            if won:
                return
            if rnd > 600:
                raise "help"

import timeit
game = SkyjoGame(2)
for games in range(2):
    test_game(game)
print(timeit.timeit('test_game(game)', globals=globals(), number=1000))
print("done")