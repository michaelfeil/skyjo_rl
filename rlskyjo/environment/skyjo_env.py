from typing import List

import numpy as np
from gym import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from rlskyjo.game.skyjo import SkyjoGame

DEFAULT_CONFIG = {
    "num_players": 3,
    "score_penalty": 2.0,
    "observe_other_player_indirect": True,
    "mean_reward": 1.0,
    "reward_refunded": 0.001,
}


def env(**kwargs):
    """wrap SkyJoEnv in"""
    env = SimpleSkyjoEnv(**kwargs)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class SimpleSkyjoEnv(AECEnv):

    metadata = {
        "render.modes": ["human"],
        "name": "skyjo",
        "is_parallelizable": False,
        "video.frames_per_second": 1,
    }

    def __init__(
        self,
        num_players=2,
        score_penalty: float = 2.0,
        observe_other_player_indirect: bool = False,
        mean_reward: float = 1.0,
        reward_refunded: float = 0.0,
    ):
        """
        Pettingzoo Gym for the card game SkyJo

        params:
            # game configuration
            num_players: int, number of players
            score_penalty: float, game default is 2.0
                score penalty for players ending but not winning the game

            # observation space configuration
            observe_other_player_indirect: bool
                True: observation space is:
                    game statistics (pile +  player cards): + own 12 player cards
                False: observation space is:
                    game statistics (excluding player cards)+ player cards of every player

            # rewards
            mean_reward: float, default: 1.0
                mean reward at the end of an game
                recommended to be > 0, e.g. Environments (like RLLib) are positive sum games
            reward_refunded: float, default: 0.0
                adds an additional reward to learn the concept of refunding cards in skyjo

        observation space is DictSpace:
            observations:
                (1,) lowest sum of players, calculated feature
                (1,) lowest number of unmasked cards of any player, calculated feature
                (15,) counts of cards past discard pile cards & open player cards,
                    calculated feature
                (1,) top discard pile card
                (1,) current hand_card
                total: (19,)

                if observe_other_player_indirect is True:
                    # constant for any num_players
                    (12) own cards
                    total: (31,)
                elif observe_other_player_indirect is False:
                    (num_players*4*3,)
                    total: (19+12*num_players,)

            action_mask:
                (26,)

        action_space is Discrete(26):
            0-11: place hand card to position 0-11
            12-23: discard place hand card and reveal position 0-11
            24: pick hand card from drawpile
            25: pick hand card from discard pile

        """
        super().__init__()

        # Hyperparams
        self.num_players = num_players
        self.mean_reward = mean_reward
        self.reward_refunded = reward_refunded

        self.table = SkyjoGame(
            num_players,
            score_penalty=score_penalty,
            observe_other_player_indirect=observe_other_player_indirect,
        )

        # start PettingZoo API stuff
        self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]

        self.agent_selection = self._expected_agentname_and_action()[0]

        self.dones = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = {i: {} for i in self.agents}

        # start obs / actions space
        self._observation_spaces = self._convert_to_dict(
            [
                spaces.Dict(
                    {
                        "observations": spaces.Box(
                            low=-24,
                            high=127,
                            shape=self.table.obs_shape,
                            dtype=self.table.card_dtype,
                        ),
                        "action_mask": spaces.Box(
                            low=0,
                            high=1,
                            shape=self.table.action_mask_shape,
                            dtype=np.int8,
                        ),
                    }
                )
                for _ in self.possible_agents
            ]
        )
        self._action_spaces = self._convert_to_dict(
            [
                spaces.Discrete(self.table.action_mask_shape[0])
                for _ in self.possible_agents
            ]
        )
        # end obs / actions space
        # end PettingZoo API stuff

    def observation_space(self, agent):
        """part of the PettingZoo API"""
        return self._observation_spaces[agent]

    def action_space(self, agent):
        """part of the PettingZoo API"""
        return self._action_spaces[agent]

    def observe(self, agent) -> dict:
        """
        get observation and action mask from environment
        part of the PettingZoo API
        """
        obs, action_mask = self.table.collect_observation(
            self._name_to_player_id(agent)
        )
        return {"observations": obs, "action_mask": action_mask}

    def step(self, action: int) -> None:
        """
        action is number from 0-25:
            0-11: place hand card to position 0-11
            12-23: discard place hand card and reveal position 0-11
            24: pick hand card from drawpile
            25: pick hand card from discard pile
        part of the PettingZoo API
        """
        current_agent = self.agent_selection
        player_id = self._name_to_player_id(current_agent)

        # if was done before
        if self.dones[current_agent]:
            return self._was_done_step(action)

        game_is_over = self.table.act(player_id, action_int=action)
        # prepare for next agent
        self.agent_selection = self._expected_agentname_and_action()[0]

        # action done, rewards if game over
        if game_is_over:
            # current player has terminated the game for all. gather rewards
            self.rewards = self._convert_to_dict(
                self._calc_final_rewards(**(self.table.get_game_metrics()))
            )
            self.dones = {i: True for i in self.agents}

        # done
        self._accumulate_rewards()
        self._clear_rewards()
        self._dones_step_first()

    def reset(self):
        """reset the environment
        part of the PettingZoo API"""
        self.table.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._expected_agentname_and_action()[0]
        self.rewards = self._convert_to_dict([0 for _ in range(self.num_agents)])
        self._cumulative_rewards = self._convert_to_dict(
            [0 for _ in range(self.num_agents)]
        )
        self.dones = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = {i: {} for i in self.agents}

    def render(self, mode="human"):
        """render board of the game

        part of the PettingZoo API"""
        if mode == "human":
            print(self.table.render_table())

    def close(self):
        """part of the PettingZoo API"""
        pass

    def seed(self, seed=None):
        """seed the environment.
         does not affect global np.random.seed()
        part of the PettingZoo API"""
        raise NotImplementedError("Seed is currently not supported with SkyJoEnv")
        
        if seed is not None:
            self.table.set_seed(seed)

    # start utils
    def _calc_final_rewards(
        self, final_score: List[int], num_refunded: List[int], **kwargs
    ):
        """
        get reward from score.
        reward is relative performance to average score
        mean reward is 1

        args:
            game_results: dict['str': np.array of len(players) e.g. np.array([35,65,50])

        returns:
            reward: np.array [len(players)] e.g. np.array([ 16,-14,+1])
        """
        score = np.array(final_score)
        reward = -score + np.mean(score) + self.mean_reward

        if self.reward_refunded:
            reward += np.array(num_refunded) * self.reward_refunded
        return reward

    @staticmethod
    def _name_to_player_id(name: str) -> int:
        """convert agent name to int  e.g. player_1 to int(1)"""
        return int(name.split("_")[-1])

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _expected_agentname_and_action(self):
        """not part of the api, implemented, get next player name for action from skyjo"""
        a = self.table.expected_action
        return f"player_{a[0]}", a[1]

    # end utils
