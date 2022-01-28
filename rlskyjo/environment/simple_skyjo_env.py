from lib2to3.pytree import Base
import numpy as np
from rlskyjo.game.skyjo_game import SkyjoGame
from gym import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
import warnings


def env(**kwargs):
    env = SimpleSkyjoEnv(**kwargs)
    # env = wrappers.CaptureStdoutWrapper(env)
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class SimpleSkyjoEnv(AECEnv):
    def __init__(self, name, num_players):
        super().__init__()
        self.name = name
        self.num_players = num_players
        self.metadata = {"render.modes": "ansi"}

        self.env = SkyjoGame(num_players)
        self.screen = None

        self.agents = [f"draw_player_{i}" for i in range(num_players)] + [
            f"place_player_{i}" for i in range(num_players)
        ]
        self.possible_agents = self.agents[:]

        # actions space
        sample_obs, sample_action_mask = self._sample_observation()

        self._dtype = self.env.card_dtype

        self.observation_spaces = self._convert_to_dict(
            [
                spaces.Dict(
                    {
                        "observations": spaces.Box(
                            low=-24,
                            high=np.iinfo(self._dtype).max,
                            shape=sample_obs.shape,
                            dtype=self._dtype,
                        ),
                        "action_mask": spaces.Box(
                            low=0, high=1, shape=sample_action_mask.shape, dtype=np.int8
                        ),
                    }
                )
                for agent_name in self.possible_agents
            ]
        )
        self.action_spaces = self._convert_to_dict(
            [
                spaces.Discrete(sample_action_mask.shape[0])
                for agent_name in self.possible_agents
            ]
        )
        # actions space done

        self.infos = {name: None for name in self.possible_agents}

        self.reset()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        config = {
            "allow_step_back": False,
            "seed": seed,
            "game_num_players": self.num_players,
        }
        self.env = SkyjoGame(self.num_players)

    def _scale_rewards(self, reward):
        return reward

    def _name_to_player_id(self, name):
        return int(name.split("_")[-1])

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _expected_agent_name(self):
        """self implemented, get next player name for action from skyjo"""
        a = self.env.expected_action
        return f"{a[1]}_player_{a[0]}"

    def _sample_observation(self):
        """self implemented to get an overview about the env"""
        playerid = self._name_to_player_id(self._expected_agent_name())
        sample_obs, action_mask = self.env.collect_observation(playerid)
        self.env.reset()
        return sample_obs, action_mask

    def observe(self, agent):
        obs, action_mask = self.env.collect_observation(self._name_to_player_id(agent))
        return {"observations": obs, "action_mask": action_mask}

    def step(self, action):
        # print(f"step {action}")
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        player_id = self._name_to_player_id(self.agent_selection)
        if self.agent_selection.startswith("draw"):
            if not 12 <= action <= 13:
                warnings.warn(f"invalid draw action: {action} not in 12-13")
                action = np.random.randint(12, 13)
            from_drawpile = bool(action)
            game_is_over, rewards_final = self.env.action_draw_card(
                player_id, from_drawpile=from_drawpile
            )
        else:
            if not 0 <= action <= 11:
                warnings.warn(f"invalid place action: {action} not in 0-11")
                action = np.random.randint(0, 11)
            game_is_over, rewards_final = self.env.action_place(
                player_id, place_to_pos=action
            )

        if game_is_over:
            self.rewards = self._convert_to_dict(
                self._scale_rewards(
                    np.repeat(rewards_final, 2)
                )  # map rewards from 1 player_id to 2 agents
            )
            # once either play wins or there is a draw, game over, both players are done
            self.dones = {i: True for i in self.agents}

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._expected_agent_name()
        self._accumulate_rewards()
        self._dones_step_first()

    def reset(self):
        self.env.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._expected_agent_name()
        self.rewards = self._convert_to_dict([0 for _ in range(self.num_agents)])
        self._cumulative_rewards = self._convert_to_dict(
            [0 for _ in range(self.num_agents)]
        )
        self.dones = self._convert_to_dict([False for _ in range(self.num_agents)])

    def render(self, mode="ansi"):
        if mode == "ansi":
            print(self.env.render_board())
        else:
            raise NotImplementedError()

    def close(self):
        pass


if __name__ == "__main__":
    rlc = SimpleSkyjoEnv("game", 2)
