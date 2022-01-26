import numpy as np
import numpy.ma as ma
import collections
import itertools
from typing import Tuple
import numba
# from numpyprint import np_format

class SkyjoGame(object):
    def __init__(self, num_players: int = 2) -> None:
        """"""
        assert 0 < num_players <= 12, "can be played up to 6/theoretical 12 players"
        # init objects
        self.num_players = num_players
        self.game_terminated = []
        
        
        # hardcoded params
        self.fill_masked_unk_value = 15
        self.fill_masked_replaced_value = 14
        self.card_dtype = np.int8
        self._name_draw = "draw"
        self._name_place = "place"
        
        # reset
        self.reset()
        
    def reset(self):       
        # 150 cards from -2 to 12
        self.game_terminated = []
        self.holding_card = self.fill_masked_unk_value
        drawpile = np.repeat(np.arange(-2,13, dtype=self.card_dtype), 10)
        np.random.shuffle(drawpile)
        self.players_cards = drawpile[:12*self.num_players].reshape(self.num_players,-1)
        
        self.drawpile = collections.deque(drawpile[self.num_players*12:])
        
        # discard_pile: first in last out
        self.discard_pile = collections.deque([self.drawpile.pop()])
        
        self.players_masked = self.reset_card_mask(self.num_players, self.card_dtype)
        self.reset_start_player()
        assert self.expected_action[1] == self._name_draw, "expect to draw after reset"

    def collect_observation(self, player_id: int):
        stats_counts, min_cards_sum, min_n_hidden, top_discard = self.observe_global_game_stats()
        player_obs = self.observe_player(player_id)
        
        obs = np.concatenate((
            player_obs.flatten(),
            [min(min_cards_sum, np.iinfo(self.card_dtype).max)],
            [min_n_hidden],
            stats_counts.flatten(),
            [top_discard],
            [self.holding_card],
        ), dtype=self.card_dtype)
        assert obs.shape == (31,)
        action_mask =  self._jit_action_mask(self.players_masked, player_id, self.expected_action[1])
        assert action_mask.shape == (14,)
        return obs, action_mask
    
    def collect_current_card(self):
        return self.holding_card
        
    @staticmethod
    @numba.njit(fastmath=True)
    def reset_card_mask(num_players, card_dtype):
        players_masked = np.full((num_players,12), 2, dtype=card_dtype)
        for pl in range(num_players):  
            picked = np.random.choice(12, 2, replace=False)
            players_masked[pl][picked] = 1
        return players_masked
    
    def reset_start_player(self):
        random_starter = np.random.randint(0, self.num_players) * 2
        assert random_starter % 2 == 0
        self.actions = itertools.cycle(([action, player]
            for action in range(self.num_players)
            for player in [self._name_draw, self._name_place]
            ))
    
        # forward to random point
        for _ in range(1 + random_starter): 
            self.next_action()
        
    def next_action(self):
        self.expected_action = next(self.actions)
        
    @staticmethod
    @numba.njit(fastmath=True)
    def _jit_observe_global_game_stats(players_cards: np.ndarray, players_masked: np.ndarray, pile: np.ndarray) -> np.ndarray:
        """mask of visible player cards
        
        return:
            flattened array
        """
        # bincount
        counted = np.array(list(pile) + list(range(-2,13)), dtype=players_cards.dtype)
        known_cards_sum = numba.typed.List()
        count_hidden = numba.typed.List()
        
        masked_option = players_masked == 1
        for pl in range(players_cards.shape[0]):
            cards_pl = players_cards[pl][masked_option[pl]]
            counted = np.concatenate((
                counted, cards_pl
            ))
            # player sums
            known_cards_sum.append(cards_pl.sum() + 0)

        counts = np.bincount(counted - np.min(counted)) - 1
        # not unknown 
        masked_option_hidden = players_masked == 2
        for pl in range(masked_option_hidden.shape[0]):
            count_hidden.append(np.sum(masked_option_hidden[pl]) + 0)
            
        known_cards_sum = np.array(list(known_cards_sum))
        count_hidden = np.array(list(count_hidden))
        return counts, np.min(known_cards_sum), np.min(count_hidden), pile[-1]
    
    @staticmethod
    @numba.njit(fastmath=True)
    def _jit_action_mask(players_masked: np.ndarray, player_id: int, next_action: str):
        if next_action == "place":
            mask_place = (players_masked[player_id] != 0).astype(np.int8)
            mask_draw = np.zeros(2, dtype=np.int8)
        else:
            mask_place = np.zeros(players_masked[player_id].shape, dtype=np.int8)
            mask_draw = np.ones(2, dtype=np.int8)
        return np.concatenate((mask_place, mask_draw))
    
    @staticmethod
    @numba.njit(fastmath=True)
    def _mask_cards_jit(players_cards, players_masked, option: int, fill = np.nan, player_id: int = None):
        """mask of visible player cards
        
        option: int 
            [0]: except replaced masked 
            [0,2]: only noe replaced & non unknown
            [2]: except unknown masked
        """
        if player_id is not None:
            players_cards = players_cards[player_id:player_id+1]
        # fill dummy value
        cards = np.full_like(
            players_cards, fill
        )
        # replace thos cards known with the actual value
        masked_notoption = players_masked != option
        for pl in range(cards.shape[0]):
            cards[pl][masked_notoption[pl]] = \
                players_cards[pl][masked_notoption[pl]]
                           
        return cards

    def observe_player(self, player_id):
        return self._mask_cards_jit(
            self.players_cards, self.players_masked,
            option = 2, fill=self.fill_masked_unk_value, player_id=player_id
        )
        
    def observe_global_game_stats(self):
        """observe game statistics, features to percieve global game"""
        # map all ever visible cards              
        return self._jit_observe_global_game_stats(
                self.players_cards,
                self.players_masked,
                np.array(self.discard_pile, dtype=self.players_cards.dtype)
        )
    
    def draw_card(self, player_id: int, from_drawpile: bool):
        """draw one card from the"""
        # assert len(self.discard_pile) + len(self.drawpile) == 150 - 12*self.num_players
        assert self.expected_action[0] == player_id and self.expected_action[1] == self._name_draw, \
            f"expected action is {self.expected_action}, but requested was [{player_id}, draw]"
        self.next_action()
        if from_drawpile:
            if not self.drawpile:
                self.reshuffle_discard_pile()
            self.holding_card= self.drawpile.pop()
            return self.holding_card
        else:
            # discard pile cannot go empty by definition
            self.holding_card = self.discard_pile.pop()
            return self.holding_card
        
    def reshuffle_discard_pile(self):
        """reshuffle discard pile into drawpile"""
        drawpile = np.array(self.discard_pile, dtype=self.card_dtype)
        np.random.shuffle(drawpile)
        self.drawpile = collections.deque(drawpile)
        self.discard_pile = collections.deque(
            [self.drawpile.pop()]
        )
    
    def play_player(self, player_id: int, place_to_pos: int) -> Tuple[bool, np.ndarray]:   
        """
        place_to_pos: int between 0 and 11, 0-11 pos, 
        
        returns:
            terminiate_game: bool, True if terminates
            winner_id
        """
        assert self.expected_action[0] == player_id and self.expected_action[1] == self._name_place, \
            f"expected action is {self.expected_action}, but requested was [{player_id}, place]"
        self.next_action()
        
        # perform goal check
        if self.game_terminated:
            return True, self.game_terminated
        game_done = self._player_goal_check(self.players_masked, player_id)
        if game_done:
            self.game_terminated = self._evaluate_game(self.players_cards, player_id)
            return True, self.game_terminated      
        
        # replace with one of the 0-11 cards of player
        # unmask new card
        # discard replaced card
        self.discard_pile.append(
            self.players_cards[player_id,place_to_pos]
        )
        self.players_masked[player_id,place_to_pos] = 1
        self.players_cards[player_id,place_to_pos] = self.holding_card
        self.holding_card = self.fill_masked_unk_value
        return False, {}
    
    @staticmethod
    @numba.njit(fastmath=True)
    def _player_goal_check(players_masked, player_id):        
        return np.all((players_masked[player_id] != 2))
    
    @staticmethod
    @numba.njit(fastmath=True)
    def _evaluate_game(players_cards, player_won_id, score_penalty = 2):       
        # reshape to size of game
        
        score = [0] * players_cards.shape[0]
        for pl in range(players_cards.shape[0]):
            for stack in range(players_cards.shape[1] // 3):
                stack_3_tup = players_cards[pl][stack*3:3+stack*3]
                
                # count penalty, when not all 3 are equal
                if np.min(stack_3_tup) != np.max(stack_3_tup):
                    score[pl] -= np.sum(stack_3_tup)
                                        
        # penalty if finisher is not winner.
        if max(score) != score[player_won_id]:
            score[player_won_id] *= score_penalty
        return score
    
    def render_game_state(self):
        str_state = \
            f"{'='*7} stats {'='*12} \n" \
            f"next turn: {self.expected_action[1]} by Player {self.expected_action[0]} \n" \
            f"holding card player {self.expected_action[0]}: {self.holding_card} \n" \
            f"discard pile top {self.discard_pile[-1]} \n" 
        return str_state
        
    def render_board(self):
        str_board = f"{'='*7} render board: {'='*5} \n"
        str_board += self.render_game_state()
        # eval_game = self._evaluate_game(self.players_cards, player_won_id=0, score_penalty=1)
        for pl in range(self.num_players):
            str_board += self.render_player(pl)
        return str_board
    
    def _render_player_cards(self, player_id):    
        array = self.players_cards[player_id].astype("str")
                    
        array[self.players_masked[player_id] == 2] = "x"
        array[self.players_masked[player_id] == 0] = "d"
        array = array.reshape(3, -1)
        array = np.array2string(
            array,
            separator='\t', 
            formatter={
                "str_kind": lambda x: str(x)
            }
        ).replace("[["," [").replace("]]","] ")
        return array

    
    def render_player(self, player_id):
        str_pl = \
            f"{'='*7} Player {player_id} {'='*10} \n" \
            f"{self._render_player_cards(player_id)} \n"
        return str_pl
    
        
    def testrun(self):
        for _ in range(5000):
            from_pile = True
            rnd = 0
            self.reset()
            won = False
            while not won:
                rnd += 1
                player_id, action = self.expected_action
                if self.game_terminated:
                    break
                if action == "draw":
                    from_pile = not from_pile 
                    observations, action_mask = self.collect_observation(player_id)
                    drawn_card = self.draw_card(player_id, from_pile)
                else:
                    position = np.random.randint(0,12)
                    won, stats = self.play_player(player_id,  position)
                    if won:
                        print(self.render_board())
                    
    

if __name__ == "__main__":
    game = SkyjoGame(2).testrun()

# print(timeit.timeit('test_game(game)', globals=globals(), number=1000))
