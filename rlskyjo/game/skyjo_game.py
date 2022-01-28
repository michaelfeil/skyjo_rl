import numpy as np
import itertools
from typing import List, Tuple
import math
import warnings

# use numba as optional dependency
try:
    from numba import njit
except ImportError:
    def njit(fastmath):
        def decorator(func):
            return func 
        return decorator



class SkyjoGame(object):
    def __init__(self, num_players: int = 2, score_penalty = 2) -> None:
        """ """
        assert 0 < num_players <= 12, "can be played up to 6/theoretical 12 players"
        # init objects
        self.num_players = num_players
        self.game_terminated = []
        self.score_penalty = score_penalty
        self.obs_shape = None

        # hardcoded params
        self.fill_masked_unk_value = 15
        self.fill_masked_refunded_value = 14
        self.card_dtype = np.int8
        self._name_draw = "draw"
        self._name_place = "place"
        
        # reset
        self.reset()

    # [start: reset utils]
    def reset(self):
        # 150 cards from -2 to 12
        self.game_terminated = []
        self.hand_card = self.fill_masked_unk_value
        drawpile = np.repeat(np.arange(-2, 13, dtype=self.card_dtype), 10)
        np.random.shuffle(drawpile)
        self.players_cards = drawpile[: 12 * self.num_players].reshape(
            self.num_players, -1
        )

        self.drawpile, self.discard_pile = self._reshuffle_discard_pile(
            drawpile[self.num_players * 12 :]
        )

        # discard_pile: first in last out
        
        self.players_masked = self.reset_card_mask(self.num_players, self.card_dtype)
        self.reset_start_player()
        assert self.expected_action[1] == self._name_draw, "expect to draw after reset"

    @staticmethod
    @njit(fastmath=True)
    def reset_card_mask(num_players, card_dtype):
        players_masked = np.full((num_players, 12), 2, dtype=card_dtype)
        for pl in range(num_players):
            picked = np.random.choice(12, 2, replace=False)
            players_masked[pl][picked] = 1
        return players_masked

    def reset_start_player(self):
        random_starter = np.random.randint(0, self.num_players) * 2
        assert random_starter % 2 == 0
        self.actions = itertools.cycle(
            (
                [action, player]
                for action in range(self.num_players)
                for player in [self._name_draw, self._name_place]
            )
        )

        # forward to random point
        for _ in range(1 + random_starter):
            self.next_action()
    
    @staticmethod
    @njit(fastmath=True)
    def _reshuffle_discard_pile(old_pile) -> Tuple[List[int],List[int]]:
        """reshuffle discard pile into drawpile.

        old_pile -> shuffle as new drawpile
        new discard_pile draw 1 card from drawpile
        """
        np.random.shuffle(old_pile)
        drawpile = list(old_pile)
        discard_pile = list([drawpile.pop()])
        return drawpile, discard_pile
    
    # [end: reset utils]

    def next_action(self):
        self.expected_action = next(self.actions)

    # [start: collect observation]

    def collect_observation(self, player_id: int):
        (
            stats_counts,
            min_cards_sum,
            min_n_hidden,
            top_discard,
        ) = self.observe_global_game_stats()
        player_obs = self.observe_player(player_id)

        obs = np.concatenate(
            (
                player_obs,
                [min(min_cards_sum, np.iinfo(self.card_dtype).max)],
                [min_n_hidden],
                stats_counts,
                [top_discard],
                [self.hand_card],
            ),
            dtype=self.card_dtype,
        )
        if self.obs_shape is None:
            self.obs_shape = obs.shape
        else:
            assert obs.shape == self.obs_shape
            
            
        action_mask = self._jit_action_mask(
            self.players_masked, player_id, self.expected_action[1]
        )
        return obs, action_mask

    @staticmethod
    @njit(fastmath=True)
    def _jit_action_mask(players_masked: np.ndarray, player_id: int, next_action: str):
        if next_action == "place":
            # must be either a card that is front of player
            mask_place = (players_masked[player_id] != 0).astype(np.int8)
            # discard hand card and reveal an masked card
            mask_place2 = (players_masked[player_id] == 2).astype(np.int8)
            mask_draw = np.zeros(2, dtype=np.int8)
        else: # draw
            # only draw allowed
            mask_place = np.zeros(players_masked[player_id].shape, dtype=np.int8)
            mask_place2 = np.zeros(players_masked[player_id].shape, dtype=np.int8)
            mask_draw = np.ones(2, dtype=np.int8)
        action_mask = np.concatenate((mask_place, mask_place2, mask_draw))
        assert action_mask.shape == (26,), "action mask needs to have shape (26,)"
        return action_mask

    def observe_global_game_stats(self) -> Tuple[np.ndarray, int, int, int]:
        """observe game statistics, features to percieve global game"""
        # map all ever visible cards
        return self._jit_observe_global_game_stats(
            self.players_cards,
            self.players_masked,
            np.array(self.discard_pile, dtype=self.players_cards.dtype),
        )

    @staticmethod
    @njit(fastmath=True)
    def _jit_observe_global_game_stats(
        players_cards: np.ndarray, players_masked: np.ndarray, pile: np.ndarray
    ) -> Tuple[np.ndarray, int, int, int]:
        """observe game statistics, features to percieve global game"""
        # bincount
        counted = np.array(list(pile) + list(range(-2, 13)), dtype=players_cards.dtype)
        known_cards_sum = [0] * players_cards.shape[0]
        count_hidden = [0] * players_cards.shape[0]

        masked_option = players_masked == 1
        for pl in range(players_cards.shape[0]):
            cards_pl = players_cards[pl][masked_option[pl]]
            counted = np.concatenate((counted, cards_pl))
            # player sums
            known_cards_sum[pl] = cards_pl.sum() + 0

        counts = np.bincount(counted - np.min(counted)) - 1
        # not unknown
        masked_option_hidden = players_masked == 2
        for pl in range(masked_option_hidden.shape[0]):
            count_hidden[pl] = np.sum(masked_option_hidden[pl]) + 0

        pile_top = pile[-1] if len(pile) else -3
        known_cards_sum = np.array(list(known_cards_sum))
        count_hidden = np.array(list(count_hidden))
        return counts.flatten(), np.min(known_cards_sum), np.min(count_hidden), pile_top

    def observe_player(self, player_id):
        return self._jit_known_player_cards(
            self.players_cards,
            self.players_masked,
            fill_unknown=self.fill_masked_unk_value,
            fill_refunded=self.fill_masked_refunded_value,
            player_id=player_id,
        )

    @staticmethod
    @njit(fastmath=True)
    def _jit_known_player_cards(
        players_cards,
        players_masked,
        fill_unknown=np.nan,
        fill_refunded=np.nan,
        player_id: int = None,
    ) -> np.array:
        """
        get array of player cards, with refunded and unknown masked with value
        """
        if player_id is not None:
            players_cards = players_cards[player_id : player_id + 1]
        # fill dummy value, assuming all cards are unknown
        cards = np.full_like(players_cards, fill_unknown)
        # replace thos cards known with the actual value
        masked_revealed = players_masked == 1
        for pl in range(cards.shape[0]):
            cards[pl][masked_revealed[pl]] = players_cards[pl][masked_revealed[pl]]

        # replace cards which are refunded with refunded value
        masked_refunded = players_masked == 0
        for pl in range(cards.shape[0]):
            cards[pl][masked_refunded[pl]] = fill_refunded

        return cards.flatten()

    # [end: collect observation]

    # [start: perform actions]
    
    def act(self, player_id: int, action_int: int):
        """perform actions"""
        assert (
            self.expected_action[0] == player_id
        ), f"ILLEGAL ACTION: expected {self.expected_action[0]}, but requested was {player_id}"
        assert 0 <= action_int <= 25, f"action int {action_int} not in range(0,26)"

        if self.game_terminated:
            warnings.warn(
                "Attemp playing terminated game. game has been already terminated by pervios player."
            )
            return True, self.game_terminated
        
        # prepare for next action
        self.next_action()
        
        if 24 <= action_int <= 25:
            assert self.hand_card == self.fill_masked_unk_value, \
                f"ILLEGAL ACTION. requested draw action {self.render_action_explainer(action_int)}" \
                f"already have a hand card {self.hand_card} "
            return self.action_draw_card(player_id, action_int)
        else:
            assert self.hand_card != self.fill_masked_unk_value, \
                f"ILLEGAL ACTION. requested place action " \
                f"but not having a hand card {self.hand_card} "
            return self.action_place(player_id, action_int)
        
    def action_draw_card(self, player_id: int, draw_from: int):
        """
        args:
            player_id: int, player who is playing
            from_drawpile: bool, action: True to draw from drawpile, else discard pile

        returns:
            game over: bool winner_id
            final_scores: list(len(n_players)) if game over
        """
        # perform goal check, games end if any player has a open 12-card deck before picking up card.
        
        game_done = self._player_goal_check(self.players_masked, player_id)
        if game_done:
            self.game_terminated = self._evaluate_game(self.players_cards, player_id, score_penalty=self.score_penalty)
            return True, self.game_terminated

        # goal is not reached. continue drawing action
        if draw_from == 24:
            # draw from drawpile
            if not self.drawpile:
                # cardpile is empty, reshuffle.
                self.drawpile, self.discard_pile = self._reshuffle_discard_pile(
                    np.array(self.discard_pile, dtype=self.card_dtype)
                )
            self.hand_card = self.drawpile.pop()
        else:
            # draw from discard pile
            # discard pile cannot go empty by definition
            self.hand_card = self.discard_pile.pop()
        # action done
        return False, []


    def action_place(self, player_id: int, action_place_to_pos: int) -> Tuple[bool, np.ndarray]:
        """
        args:
            player_id: int, player who is playing
            action_place_to_pos: int, action between 0 and 11,

        returns:
            game over: bool winner_id
            final_scores: list(len(n_players)) if game over
        """
        
        if action_place_to_pos in range(0,12):
            # replace one of the 0-11 cards with hand card of player
            # unmask new card
            # discard deck card
            self.discard_pile.append(self.players_cards[player_id, action_place_to_pos])
            self.players_masked[player_id, action_place_to_pos] = 1
            self.players_cards[player_id, action_place_to_pos] = self.hand_card
        else: 
            # discard hand card, reveal a yet unrevealed card
            place_pos = action_place_to_pos - 12
            assert self.players_masked[player_id, place_pos] == 2, \
                f"illegal action {self.render_action_explainer(action_place_to_pos)}."\
                f"card is already revealed: {self.players_masked[player_id]}"
            self.discard_pile.append(self.hand_card)
            self.players_masked[player_id, place_pos] = 1
        
        self.hand_card = self.fill_masked_unk_value
        return False, []

    # [end: perform actions]

    @staticmethod
    @njit(fastmath=True)
    def _player_goal_check(players_masked, player_id):
        """check if game over, when player_id has all cards known (=!2)"""
        return np.all((players_masked[player_id] != 2))

    @staticmethod
    @njit(fastmath=True)
    def _evaluate_game(players_cards, player_won_id, score_penalty: float = 2.0) -> List[int]:
        """
        calculate game scores
        """

        score = [0.0] * players_cards.shape[0]
        for pl in range(players_cards.shape[0]):
            for stack in range(players_cards.shape[1] // 3):
                stack_3_tup = players_cards[pl][stack * 3 : 3 + stack * 3]

                # count penalty, when not all 3 are equal
                if np.min(stack_3_tup) != np.max(stack_3_tup):
                    score[pl] += np.sum(stack_3_tup)

        # penalty if finisher is not winner.
        if min(score) != score[player_won_id]:
            score[player_won_id] *= score_penalty
        return score

    # [start: render utils]

    def render_table(self):
        """
        render game:
            render cards for all players
            render game statistics
        """
        render_cards_open = False
        str_board = f"{'='*7} render board: {'='*5} \n"
        str_board += self.render_game_stats()
        if self.game_terminated:
            str_board += (
                f"{'='*7} GAME DONE {'='*8} \n"
                f"Results: {dict(zip(list(range(self.num_players)), self.game_terminated))} \n"
            )
            render_cards_open = True
        for pl in range(self.num_players):
            str_board += self.render_player(pl, render_cards_open)
        return str_board
    
    def render_game_stats(self):
        """render game statistics"""
        card_hand = self.hand_card if -2 <= self.hand_card <= 12 else "empty"
        discard_pile_top = self.discard_pile[-1] if self.discard_pile else "empty"
        str_stats = (
            f"{'='*7} stats {'='*12} \n"
            f"next turn: {self.expected_action[1]} by Player {self.expected_action[0]} \n"
            f"holding card player {self.expected_action[0]}: "
            f"{card_hand} \n"
            f"discard pile top: {discard_pile_top} \n"
        )

        return str_stats

    def _render_player_cards(self, player_id, render_cards_open):    
        array = self.players_cards[player_id].astype(np.str_)
        
        if render_cards_open:        
            array[self.players_masked[player_id] == 2] = np.char.add(
                np.array(["x"], dtype=np.str_),
                array[self.players_masked[player_id] == 2]
                )
        else:
            array[self.players_masked[player_id] == 2] = "x"
        
        array[self.players_masked[player_id] == 0] = "d"
        array = array.reshape(3, -1)
        array = np.array2string(
            array,
            separator='\t ', 
            formatter={
                "str_kind": lambda x: str(x)
            }
        )
        return array

    
    def render_player(self, player_id, render_cards_open=False):
        """render cards of 1 player"""
        str_pl = f"{'='*7} Player {player_id} {'='*10} \n"
        str_pl += self._render_player_cards(player_id, render_cards_open) + "\n"
        return str_pl
    
    def render_action_explainer(self, action_int: int):
        assert action_int in range(0,26), "action not valid action int {action_int}"
     
        if action_int == 24:
            return "draw from drawpile"
        elif action_int == 25:
            return "draw from discard pile"
        
        if action_int in range(0,12):
            place_id = action_int
            result = f"place card ({action_int}) - "
        elif action_int in range(12,24):
            place_id = action_int - 12
            result = f"handcard discard & reveal card ({action_int}) - "
        
        col = place_id % 4
        row = math.floor(place_id / 3)
        
        result += f"col:{col} row:{row}"
        
        return result
            
    # [end: render utils]

    # [start: test utils]

    def testrun(self) -> None:
        """perform random run to test game"""
        for _ in range(5000):
            from_pile = True
            rnd = 0
            self.reset()
            won = False
            while not self.game_terminated:
                rnd += 1
                player_id, action_expected = self.expected_action
                observations, action_mask = self.collect_observation(player_id)

                # pick a valid random action
                action = self.random_admissible_policy(observations, action_mask)
            
                # print(self.render_table())
                # print(self.render_action_explainer(action))
                
                won, stats = self.act(player_id, action)
                
                # print(self.render_table())
            else:
                # upon termination
                print(self.render_table())

    @staticmethod
    def random_admissible_policy(observation: np.array, action_mask: np.array) -> int:
        """picks randomly an admissible action from the action mask"""
        assert len(observation)
        return np.random.choice(
            np.arange(len(action_mask)), p=action_mask / np.sum(action_mask)
        )

    # [end: test utils]


if __name__ == "__main__":
    game = SkyjoGame(2).testrun()

# print(timeit.timeit('test_game(game)', globals=globals(), number=1000))
