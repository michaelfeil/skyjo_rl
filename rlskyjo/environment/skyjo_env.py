import numpy as np
from rlskyjo.game.skyjo_game import SkyjoGame
from gym import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers


def env(**kwargs):
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

    def __init__(self, num_players=2, time_penalty=0.0, score_penalty=1.2):
        super().__init__()

        # Hyperparams
        self.num_players = num_players
        self.time_penalty = time_penalty
        self._dtype = self.table.card_dtype

        self.table = SkyjoGame(num_players, score_penalty=score_penalty)

        # start PettingZoo API
        self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]

        self.agent_selection = self._expected_agentname_and_action()[0]
        self.rewards = self._convert_to_dict([0 for _ in range(self.num_agents)])
        self._cumulative_rewards = self._convert_to_dict(
            [0 for _ in range(self.num_agents)]
        )
        self.dones = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = {i: {} for i in self.agents}
        # end PettingZoo API

        # start obs / actions space
        agent = self._expected_agentname_and_action()[0]
        obs = self.observe(agent)

        self._observation_spaces = self._convert_to_dict(
            [
                spaces.Dict(
                    {
                        "observations": spaces.Box(
                            low=-24,
                            high=127,
                            shape=obs["observations"].shape,
                            dtype=self._dtype,
                        ),
                        "action_mask": spaces.Box(
                            low=0, high=1, shape=obs["action_mask"].shape, dtype=np.int8
                        ),
                    }
                )
                for _ in self.possible_agents
            ]
        )
        self._action_spaces = self._convert_to_dict(
            [spaces.Discrete(obs["action_mask"].shape[0]) for _ in self.possible_agents]
        )
        # end obs / actions space

    def observation_space(self, agent):
        """part of the PettingZoo API"""
        return self._observation_spaces[agent]

    def action_space(self, agent):
        """part of the PettingZoo API"""
        return self._action_spaces[agent]

    @staticmethod
    def _calc_final_rewards(score: np.array):
        """
        get reward from score. reward is relative performance to average score

        args:
            score: np.array of len(players) e.g. np.array([35,65,50])

        returns:
            reward: np.array [len(players)] e.g. np.array([ 16,-14,+1])
        """
        score = np.array(score)
        return -score + np.mean(score) + 1

    def _name_to_player_id(self, name: str) -> int:
        """convert agent name to int  e.g. player_1 to int(1)"""
        return int(name.split("_")[-1])

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _expected_agentname_and_action(self):
        """not part of the api, implemented, get next player name for action from skyjo"""
        a = self.table.expected_action
        return f"player_{a[0]}", a[1]

    def _sample_observation(self):
        """self implemented to get an overview about the env"""
        agent = self._expected_agentname_and_action()[0]
        obs = self.observe(agent)

        return obs

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
        action is number from 0-13:
            0-11: place hand card to position 0-11
            12-23: discard place hand card and reveal position 0-11
            24: pick hand card from drawpile
            25: pick hand card from discard pile
        part of the PettingZoo API
        """
        current_agent = self.agent_selection
        if self.dones[current_agent]:
            return self._was_done_step(action)

        player_id = self._name_to_player_id(current_agent)

        game_is_over, score_final = self.table.act(player_id, action_int=action)

        if game_is_over:
            # current player has terminated the game for all. gather rewards
            self.rewards = self._convert_to_dict(self._calc_final_rewards(score_final))
            self.dones = {i: True for i in self.agents}

        # add some penalty for not finishing
        if self.time_penalty:
            self.rewards[current_agent] -= self.time_penalty

        self.agent_selection = self._expected_agentname_and_action()[0]
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
        else:
            raise NotImplementedError()

    def close(self):
        """part of the PettingZoo API"""
        pass


if __name__ == "__main__":
    rlc = SimpleSkyjoEnv()
