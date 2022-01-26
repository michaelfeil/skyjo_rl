import numpy as np
from rlskyjo.game.skyjo_game import SkyjoGame
from gym import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers


class RLCardBase(AECEnv):
    def __init__(self, name, num_players):
        super().__init__()
        self.name = name
        self.num_players = num_players
        config = {'allow_step_back': False,
                  'seed': None,
                  'game_num_players': num_players}

        self.env = SkyjoGame(num_players)
        self.screen = None

        self.agents = [f'player_{i}' for i in range(num_players)]
        self.possible_agents = self.agents[:]

        sample_obs, sample_action_mask = self.sample_observation_action()
        
        self._dtype = np.dtype(sample_obs.dtype)

        self.observation_spaces = self._convert_to_dict(
            [spaces.Dict({'observation': spaces.Box(low=-16.0, high=16.0, shape=sample_obs.shape, dtype=self._dtype),
                          'action_mask': spaces.Box(low=0, high=1, shape=sample_action_mask.shape,
                                                    dtype=np.int8)}) for _ in range(self.num_agents)])
        self.action_spaces = self._convert_to_dict([spaces.Discrete(sample_action_mask.shape[0]) for _ in range(self.num_agents)])
    
    def sample_observation_action(self):
        sample_obs, action_mask = self.env.collect_observation_draw(0)
        card = self.env.draw_card(True)
        
        return np.append(
            sample_obs, card
        ), action_mask
                                      
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        config = {'allow_step_back': False,
                  'seed': seed,
                  'game_num_players': self.num_players}
        self.env = SkyjoGame(self.num_players)

    def _scale_rewards(self, reward):
        return reward

    def _int_to_name(self, ind):
        return self.possible_agents[ind]

    def _name_to_int(self, name):
        return self.possible_agents.index(name)

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def observe(self, agent):
        obs_w_o_card, action_mask = self.env.collect_observation_draw(self._name_to_int(agent))
        # TODO: policy to decide true false
        card = self.env.draw_card(True)
        observation = np.append(obs_w_o_card, card)

        return {'observation': observation, 'action_mask': action_mask}

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        obs, next_player_id = self.env.step(action)
        next_player = self._int_to_name(next_player_id)
        self._last_obs = self.observe(self.agent_selection)
        if self.env.is_over():
            self.rewards = self._convert_to_dict(self._scale_rewards(self.env.get_payoffs()))
            self.next_legal_moves = []
            self.dones = self._convert_to_dict([True if self.env.is_over() else False for _ in range(self.num_agents)])
        else:
            self.next_legal_moves = obs['legal_actions']
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_player
        self._accumulate_rewards()
        self._dones_step_first()

    def reset(self):
        obs, player_id = self.env.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._int_to_name(player_id)
        self.rewards = self._convert_to_dict([0 for _ in range(self.num_agents)])
        self._cumulative_rewards = self._convert_to_dict([0 for _ in range(self.num_agents)])
        self.dones = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = self._convert_to_dict([{'legal_moves': []} for _ in range(self.num_agents)])
        self.next_legal_moves = list(sorted(obs['legal_actions']))
        self._last_obs = obs['obs']

    def render(self, mode='human'):
        raise NotImplementedError()

    def close(self):
        pass

if __name__ == "__main__":
    rlc = RLCardBase("game", 2)