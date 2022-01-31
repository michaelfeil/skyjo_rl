import itertools
import math
import warnings
from typing import List, Tuple

import numpy as np

# use numba as optional dependency
try:
    from numba import njit
except ImportError:
    # in case numba is not installed -> njit functions are just python/slow.
    def njit(fastmath):
        def decorator(func):
            return func

        return decorator


class SkyjoGame(object):
    def __init__(
        self, num_players: int = 3, score_penalty=2, observe_other_player_indirect=False
    ) -> None:
        """ """
        assert (
            0 < num_players <= 12
        ), "Skyjo can be played from 1 up to 8 (recommended) / 12 (theoretical) players"

        # init objects
        self.num_players = num_players
        self.score_penalty = score_penalty

        # hardcoded params
        self.fill_masked_unk_value = 15
        self.fill_masked_refunded_value = -14
        self.card_dtype = np.int8
        self._name_draw = "draw"
        self._name_place = "place"

        # observation of other players:
        # indirect: via statistics (observation space invariant of num_players)
        # direct: add other observations to player stats
        self.observe_other_player_indirect = observe_other_player_indirect
        self.obs_shape = (
            (19 + 12,) if observe_other_player_indirect else (19 + num_players * 12,)
        )
        self.action_mask_shape = (26,)

        # reset
        self.reset()

    # [start: reset utils]
    def reset(self):
        # 150 cards from -2 to 12
        self.is_terminated = False
        # metrics
        self.game_metrics = {
            "num_refunded": [0] * self.num_players,
            "num_placed": [0] * self.num_players,
            "final_score": False,
        }
        self.hand_card = self.fill_masked_unk_value
        drawpile = self._new_drawpile(self.card_dtype)
        self.players_cards = drawpile[: 12 * self.num_players].reshape(
            self.num_players, -1
        )

        # discard_pile: first in last out
        self.drawpile, self.discard_pile = self._reshuffle_discard_pile(
            drawpile[self.num_players * 12 :]
        )

        self.players_masked = self._reset_card_mask(self.num_players, self.card_dtype)
        self._reset_start_player()
        assert self.expected_action[1] == self._name_draw, "expect to draw after reset"

    @staticmethod
    @njit()
    def _new_drawpile(card_dtype=np.int8):
        """create a drawpile len(150) and cards from -2 to 12"""
        drawpile = np.repeat(np.arange(-2, 13, dtype=card_dtype), 10)
        np.random.shuffle(drawpile)
        return drawpile

    def set_seed(self, value):
        """adds a random number generator. does not affect global np.random.seed()"""
        self.rng = np.random.default_rng(value)
        self._set_seed_njit(value + 1)
        self.reset()

    @staticmethod
    @njit()
    def _set_seed_njit(value: int):
        """set seed for numba"""
        try:
            []  # fails in numba
        except Exception:
            np.random.seed(value)

    @staticmethod
    @njit(fastmath=True)
    def _reset_card_mask(num_players, card_dtype):
        players_masked = np.full((num_players, 12), 2, dtype=card_dtype)
        for pl in range(num_players):
            picked = np.random.choice(12, 2, replace=False)
            players_masked[pl][picked] = 1
        return players_masked

    def _reset_start_player(self):
        player_counts = self._jit_observe_global_game_stats(
            self.players_cards,
            self.players_masked,
            np.array(self.discard_pile, dtype=self.players_cards.dtype),
        )[1]
        # player with most cards starts to draw
        starter_id = player_counts.argmax() * 2

        self.actions = itertools.cycle(
            (
                [action, player]
                for action in range(self.num_players)
                for player in [self._name_draw, self._name_place]
            )
        )

        # forward to point of expected action
        # where player with most cards starts to draw
        for _ in range(1 + starter_id):
            self._internal_next_action()

    @staticmethod
    @njit(fastmath=True)
    def _reshuffle_discard_pile(old_pile) -> Tuple[List[int], List[int]]:
        """reshuffle discard pile into drawpile.

        old_pile -> shuffle as new drawpile
        new discard_pile draw 1 card from drawpile
        """
        np.random.shuffle(old_pile)
        drawpile = list(old_pile)
        discard_pile = list([drawpile.pop()])
        return drawpile, discard_pile

    # [end: reset utils]

    def _internal_next_action(self):
        """set next expected action"""
        self.expected_action = next(self.actions)

    # [start: collect observation]

    def collect_observation(self, player_id: int) -> Tuple[np.array, np.array]:

        # get global stats
        (
            stats_counts,
            cards_sum,
            n_hidden,
            top_discard,
        ) = self._jit_observe_global_game_stats(
            self.players_cards,
            self.players_masked,
            np.array(self.discard_pile, dtype=self.players_cards.dtype),
            count_players_cards=not self.observe_other_player_indirect,
        )

        # get player observation
        if self.observe_other_player_indirect:
            # observe only self
            player_obs = self._jit_known_player_cards(
                self.players_cards,
                self.players_masked,
                fill_unknown=self.fill_masked_unk_value,
                player_id=player_id,
            )
        else:
            player_obs = self._jit_known_player_cards_all(
                self.players_cards,
                self.players_masked,
                fill_unknown=self.fill_masked_unk_value,
                player_id=player_id,
            )
        # concat observation
        obs = np.array(
            (
                [min(cards_sum.min(), 127)]  # (1,)
                + [n_hidden.min()]  # (1,)
                + stats_counts  # (15,)
                + [top_discard]  # (1,)
                + [self.hand_card]  # (1,)
                + player_obs  # (12,) or (num_players * 12,)
            ),
            dtype=self.card_dtype,
        )

        assert obs.shape == self.obs_shape, (
            "unexpected observation shape" f"{obs.shape} expected {self.obs_shape}"
        )

        action_mask = self._jit_action_mask(
            self.players_masked, player_id, self.expected_action[1]
        )
        return obs, action_mask

    @staticmethod
    @njit(fastmath=True)
    def _jit_action_mask(
        players_masked: np.ndarray,
        player_id: int,
        next_action: str,
        action_mask_shape=(26,),
    ):
        if next_action == "place":
            # must be either a card that is front of player
            mask_place = (players_masked[player_id] != 0).astype(np.int8)
            # discard hand card and reveal an masked card
            mask_place2 = (players_masked[player_id] == 2).astype(np.int8)
            mask_draw = np.zeros(2, dtype=np.int8)
        else:  # draw
            # only draw allowed
            mask_place = np.zeros(players_masked[player_id].shape, dtype=np.int8)
            mask_place2 = np.zeros(players_masked[player_id].shape, dtype=np.int8)
            mask_draw = np.ones(2, dtype=np.int8)
        action_mask = np.concatenate((mask_place, mask_place2, mask_draw))
        assert (
            action_mask.shape == action_mask_shape
        ), "action mask needs to have shape (26,)"
        return action_mask

    @staticmethod
    @njit(fastmath=True)
    def _jit_observe_global_game_stats(
        players_cards: np.ndarray,
        players_masked: List[int],
        pile: np.ndarray,
        count_players_cards: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """observe game statistics, features to percieve global game"""
        # bincount
        counted = np.array(list(pile) + list(range(-2, 13)), dtype=players_cards.dtype)
        known_cards_sum = [0] * players_cards.shape[0]
        count_hidden = [0] * players_cards.shape[0]

        masked_option = players_masked == 1
        for pl in range(players_cards.shape[0]):
            cards_pl = players_cards[pl][masked_option[pl]]
            if count_players_cards:
                counted = np.concatenate((counted, cards_pl))
            # player sums
            known_cards_sum[pl] = cards_pl.sum()

        counts = np.bincount(counted - np.min(counted)) - 1
        # not unknown
        masked_option_hidden = players_masked == 2
        for pl in range(masked_option_hidden.shape[0]):
            count_hidden[pl] = np.sum(masked_option_hidden[pl])

        pile_top = pile[-1] if len(pile) else -3
        known_cards_sum = np.array(known_cards_sum)
        count_hidden = np.array(count_hidden)
        return list(counts.flatten()), known_cards_sum, count_hidden, pile_top

    @staticmethod
    @njit(fastmath=True)
    def _jit_known_player_cards(
        players_cards,
        players_masked,
        player_id: int,
        fill_unknown=np.nan,
    ) -> np.array:
        """
        get array of player cards, with refunded and unknown masked with value

        return:
            array of size (4,3) or (12,) if flatten
        """
        cards = np.full_like(players_cards[player_id], fill_unknown)

        masked_revealed = players_masked[player_id] != 2
        cards[masked_revealed] = players_cards[player_id][masked_revealed]
        return list(cards.flatten())

    @staticmethod
    @njit(fastmath=True)
    def _jit_known_player_cards_all(
        players_cards,
        players_masked,
        player_id: int,
        fill_unknown=np.nan,
    ) -> np.array:
        """
        get array of player cards, with refunded and unknown masked with value

        return:
            array of size players_cards (num_players, 4, 3)
                        or players_cards.flatten() if flatten
        """
        cards = np.full_like(players_cards, fill_unknown)

        # assign in the desired order
        for player_id_iter in np.roll(np.arange(players_cards.shape[0]), player_id):
            masked_revealed = players_masked[player_id_iter] != 2
            cards[player_id_iter][masked_revealed] = players_cards[player_id_iter][
                masked_revealed
            ]
        return list(cards.flatten())

    # [end: collect observation]

    # [start: perform actions]

    def act(self, player_id: int, action_int: int):
        """perform actions"""
        assert self.expected_action[0] == player_id, (
            f"ILLEGAL ACTION: expected {self.expected_action[0]}"
            f" but requested was {player_id}"
        )
        assert 0 <= action_int <= 25, f"action int {action_int} not in range(0,26)"

        if self.is_terminated:
            warnings.warn(
                "Attemp playing terminated game."
                " game has been already terminated by pervios player."
            )
            return True

        if 24 <= action_int <= 25:
            assert self.hand_card == self.fill_masked_unk_value, (
                "ILLEGAL ACTION. requested draw action"
                f" {self.render_action_explainer(action_int)}"
                f"already have a hand card {self.hand_card} "
            )
            return self._action_draw_card(player_id, action_int)
        else:
            assert self.hand_card != self.fill_masked_unk_value, (
                f"ILLEGAL ACTION. requested place action "
                f"but not having a hand card {self.hand_card} "
            )
            return self._action_place(player_id, action_int)

    def _action_draw_card(self, player_id: int, draw_from: int):
        """
        args:
            player_id: int, player who is playing
            from_drawpile: bool, action: True to draw from drawpile, else discard pile

        returns:
            game over: bool winner_id
            final_scores: list(len(n_players)) if game over
        """
        # perform goal check
        # games end if any player has a open 12-card deck before picking up card.

        game_done = self._player_goal_check(self.players_masked, player_id)
        if game_done:
            self.is_terminated = True
            self.game_metrics["final_score"] = self._evaluate_game(
                self.players_cards, player_id, score_penalty=self.score_penalty
            )
            return True

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
        self._internal_next_action()
        return False

    def _action_place(
        self, player_id: int, action_place_to_pos: int
    ) -> Tuple[bool, np.ndarray]:
        """
        args:
            player_id: int, player who is playing
            action_place_to_pos: int, action between 0 and 11,

        returns:
            game over: bool winner_id
            final_scores: list(len(n_players)) if game over
        """

        if action_place_to_pos in range(0, 12):
            # replace one of the 0-11 cards with hand card of player
            # unmask new card
            # discard deck card
            self.discard_pile.append(self.players_cards[player_id][action_place_to_pos])
            self.players_masked[player_id][action_place_to_pos] = 1
            self.players_cards[player_id][action_place_to_pos] = self.hand_card
        else:
            # discard hand card, reveal a yet unrevealed card
            place_pos = action_place_to_pos - 12
            assert self.players_masked[player_id][place_pos] == 2, (
                f"illegal action {self.render_action_explainer(action_place_to_pos)}."
                f"card is already revealed: {self.players_masked[player_id]}"
            )
            self.discard_pile.append(self.hand_card)
            self.players_masked[player_id][place_pos] = 1

        # check if three in a row -> discard and mask cards then.
        (
            is_updated,
            pc_update,
            pm_update,
            dp_add,
        ) = self._remask_refunded_player_cards_jit(
            self.players_cards,
            self.players_masked,
            player_id,
            self.fill_masked_refunded_value,
        )
        if is_updated:
            self.game_metrics["num_refunded"][player_id] += 1
            self.players_cards, self.players_masked = pc_update, pm_update
            self.discard_pile.extend(dp_add)

        # action done
        self.game_metrics["num_placed"][player_id] += 1
        self.hand_card = self.fill_masked_unk_value
        self._internal_next_action()
        return False

    # [end: perform actions]

    @staticmethod
    @njit(fastmath=True)
    def _remask_refunded_player_cards_jit(
        players_cards,
        players_masked,
        player_id: int,
        fill_masked_refunded_value: int = -14,
    ):
        """
        check if any cards of the player got refunded.
        if so return new players_cards, players_masked and additions to discard pile
        """
        cards_to_discard_pile = np.empty((0,), dtype=np.int8)
        values_updated = False

        # for stack in [0, 1, 2, 3]
        for stack in range(players_cards[player_id].shape[0] // 3):
            slice_tup = slice(stack * 3, stack * 3 + 3, 1)
            cards_stack_3_tup = players_cards[player_id][slice_tup]
            # check if all got the same value
            if np.min(cards_stack_3_tup) == np.max(cards_stack_3_tup):
                # check if the same value has been masked before
                if np.all(players_masked[player_id][slice_tup] == 1):
                    players_masked[player_id][slice_tup] = 0

                    cards_to_discard_pile = np.append(
                        cards_to_discard_pile, players_masked[player_id][slice_tup]
                    )
                    players_cards[player_id][slice_tup] = fill_masked_refunded_value
                    values_updated = True
        if values_updated:
            return (
                values_updated,
                players_cards,
                players_masked,
                list(cards_to_discard_pile),
            )
        else:
            return values_updated, None, None, None

    @staticmethod
    @njit(fastmath=True)
    def _player_goal_check(players_masked, player_id):
        """check if game over, when player_id has all cards known (=!2)"""
        return np.all((players_masked[player_id] != 2))

    @staticmethod
    @njit(fastmath=True)
    def _evaluate_game(
        players_cards, player_won_id, score_penalty: float = 2.0
    ) -> List[int]:
        """
        calculate game scores
        """

        score = [0.0] * players_cards.shape[0]

        for pl in range(players_cards.shape[0]):
            for stack in range(players_cards[pl].shape[0] // 3):
                slice_tup = slice(stack * 3, stack * 3 + 3, 1)
                cards_stack_3_tup = players_cards[pl][slice_tup]
                if np.min(cards_stack_3_tup) != np.max(cards_stack_3_tup):
                    score[pl] += np.sum(cards_stack_3_tup)

        # penalty if finisher is not winner.
        if min(score) != score[player_won_id]:
            score[player_won_id] *= score_penalty
        return score

    def get_game_metrics(self):
        return self.game_metrics

    def get_expected_action(self):
        return self.expected_action

    # [start: render utils]

    def render_table(self):
        """
        render game:
            render cards for all players
            render game statistics
        """
        render_cards_open = False
        str_board = f"{'='*7} render board: {'='*5} \n"
        str_board += self._render_game_stats()
        if self.is_terminated:
            res = dict(
                zip(list(range(self.num_players)), self.game_metrics["final_score"])
            )
            str_board += f"{'='*7} GAME DONE {'='*8} \n" f"Results: {res} \n"
            render_cards_open = True
        for pl in range(self.num_players):
            str_board += self.render_player(pl, render_cards_open)
        return str_board

    def _render_game_stats(self):
        """render game statistics"""
        card_hand = self.hand_card if -2 <= self.hand_card <= 12 else "empty"
        discard_pile_top = self.discard_pile[-1] if self.discard_pile else "empty"
        str_stats = (
            f"{'='*7} stats {'='*12} \n"
            f"next turn: {self.expected_action[1]} "
            f"by Player {self.expected_action[0]} \n"
            f"holding card player {self.expected_action[0]}: "
            f"{card_hand} \n"
            f"discard pile top: {discard_pile_top} \n"
        )

        return str_stats

    def _render_player_cards(self, player_id, render_cards_open):
        array = self.players_cards[player_id].astype(np.str_)

        if render_cards_open:
            array[self.players_masked[player_id] == 2] = np.char.add(
                np.array(["u"], dtype=np.str_),
                array[self.players_masked[player_id] == 2],
            )
        else:
            array[self.players_masked[player_id] == 2] = "u"

        array[self.players_masked[player_id] == 0] = "d"
        array = array.reshape(4, -1).T
        array = np.array2string(
            array, separator="\t ", formatter={"str_kind": lambda x: str(x)}
        )
        return array

    def render_player(self, player_id, render_cards_open=False):
        """render cards of 1 player"""
        str_pl = f"{'='*7} Player {player_id} {'='*10} \n"
        str_pl += self._render_player_cards(player_id, render_cards_open) + "\n"
        return str_pl

    @classmethod
    def render_action_explainer(cls, action_int: int):
        """adds a string explaining actions to plot of render_player"""
        assert action_int in range(0, 26), "action not valid action int {action_int}"

        if action_int == 24:
            return "draw from drawpile"
        elif action_int == 25:
            return "draw from discard pile"

        if action_int in range(0, 12):
            place_id = action_int
            result = f"place card ({action_int}) - "
        elif action_int in range(12, 24):
            place_id = action_int - 12
            result = f"handcard discard & reveal card ({action_int}) - "

        col = math.floor(place_id / 3)

        row = place_id % 4

        result += f"col:{col} row:{row}"

        return result

    @classmethod
    def render_actions(cls):
        """possible actions"""
        array = np.char.add(np.arange(12).reshape(4, -1).T.astype(np.str_), "/")
        array = np.char.add(array, np.arange(12, 24).reshape(4, -1).T.astype(np.str_))
        array = np.array2string(
            array, separator="\t ", formatter={"str_kind": lambda x: str(x)}
        )
        return (
            f"action ids 0-25: \n(put handcard here / reveal this card) \n {array} \n"
            f"24: draw from drawpile \n 25: draw from discard pile"
        )

    # [end: render utils]


if __name__ == "__main__":
    game = SkyjoGame(2)._testrun()

# print(timeit.timeit('test_game(game)', globals=globals(), number=1000))
