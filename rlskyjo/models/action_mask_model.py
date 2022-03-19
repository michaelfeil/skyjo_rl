# adapted from https://github.com/ray-project/ray/blob/5ec63ccc5f351f143bd325e764d1f2a523023ca9/rllib/examples/models/action_mask_model.py

from gym.spaces import Dict, Box
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
import gym
import numpy as np

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version


    Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()

class TorchPlayerRelation(TorchModelV2, nn.Module):
    """PyTorch version


    Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        hidden_dim=16,
        statistics_lenght=19, # card sizes
        card_size=[4,3], # card sizes
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )
        self.statistics_lenght = statistics_lenght
        self.card_size = card_size

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.card3_conv = nn.Sequential(
            nn.Linear(card_size[1], hidden_dim),
            nn.ReLU(),
        )

        self.rows_conv = nn.Sequential(
            nn.Linear(card_size[0]*hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
        )

        self.fc_player_pairs = nn.Sequential(
            nn.Linear(hidden_dim*4+2, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim*4),
            nn.ReLU(),
        )
        self.f_norm = nn.LayerNorm(hidden_dim*4)


        self.internal_fcmodel = TorchFC(
            Box(low=-10, high=127, shape=(hidden_dim*4 + statistics_lenght,)),
            action_space,
            num_outputs,
            model_config,
            name + "_internal_out",
        )
        
        
        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def factorize_observation(self, obs):
        batch_size, obs_size = obs.size()
        flat_statistics = obs[:, :self.statistics_lenght]

        player_cards = obs[:, self.statistics_lenght:].reshape(batch_size, -1, self.card_size[0], self.card_size[1])

        return flat_statistics, player_cards

    @staticmethod
    def _add_entitiy_indices_as_feature(input_v):
        """add range over top of embedding

        Args:
            input_v (_type_): _description_

        Returns:
            _type_: _description_
        """
        orig_shape = input_v.shape
        
        features_indices = np.arange(
            orig_shape[1], # add features for indices
        )
        std = features_indices.std()
        if std:
            features_indices = (features_indices - features_indices.mean()) / std
                
        features_indices = np.tile(features_indices, (orig_shape[0],1)).reshape(orig_shape[0],orig_shape[1],-1)
        return torch.cat(
            (
                torch.from_numpy(features_indices).to(dtype=input_v.dtype, device=input_v.device),
                input_v
            ),
            dim = 2
        )

    def internal_relation(self, flat_statistics, player_cards):
        """
            flat_statistics: shape (batch, self.statistics_lenght,)
            player_cards: shape (batch, n_players, card_size[0]=4, card_size[1]=3)
            
        return:
            game_embedding shape (batch, self.statistics_lenght + self.hidden_size * 4)
        """
        shape_cards = player_cards.size()
        # player_cards_over_rows shape = (batch, n_players, card_size[0] * hidden_dim)
        player_cards_over_rows = self.card3_conv(player_cards).flatten(2)

        # player_cards_over_rows shape = (batch, n_players, 2 * hidden_dim)
        player_encoding = self.rows_conv(player_cards_over_rows)

        player_encoding = self._add_entitiy_indices_as_feature(player_encoding)

        n_players = player_encoding.size(1)
        xi = player_encoding.repeat(1, n_players, 1)
        xj = player_encoding.unsqueeze(2)
        xj = xj.repeat(1,1,n_players,1).view(player_encoding.size(0), -1, player_encoding.size(2))
        # pair_concat.size=(batch_size, n_players*n_players, 4 * hidden_dim + 2)
        # 
        pair_concat = torch.cat((xi,xj), dim=2) 
        player_relations = self.fc_player_pairs(pair_concat)
        
        embedding_player = self.f_norm(torch.sum(player_relations, dim=1))
        
        game_embedding = torch.cat((flat_statistics.flatten(1),embedding_player.flatten(1)), dim=1)

        # re
        return game_embedding
    
    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]
        observation = input_dict["obs"]["observations"]
        unsqueezed = False
        if observation.ndim == 1:
            unsqueezed = True
            observation = torch.unsqueeze(observation, 0)

        flat_statistics, player_cards = self.factorize_observation(observation)
        # Compute the unmasked logits.
        game_embedding = self.internal_relation(flat_statistics, player_cards)
        logits, _ = self.internal_fcmodel({"obs": game_embedding})
        # If action masking is disabled, directly return unmasked logits
        if unsqueezed:
            logits = torch.squeeze(logits)
        
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_fcmodel.value_function()
