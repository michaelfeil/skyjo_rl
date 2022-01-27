from copy import deepcopy
import os
import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import PettingZooEnv
from rlskyjo.environment import simple_skyjo_env
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from gym.spaces import Box
from ray.rllib.utils.framework import try_import_torch
from rlskyjo.models.action_mask_model import TorchMaskedActions

torch, nn = try_import_torch()
ray.init(local_mode=True)


    
if __name__ == "__main__":
    alg_name = "DQN"
    env_name  = "pettingzoo_skyjo"
    ModelCatalog.register_custom_model(
        "pa_model", TorchMaskedActions
    )
    # function that outputs the environment you wish to register.

    def env_creator():
        env = simple_skyjo_env.env(**{"name":env_name, "num_players": 2})
        return env


    config = deepcopy(get_agent_class(alg_name)._default_config)

    register_env(env_name,
                 lambda config: PettingZooEnv(env_creator()))

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    print("obs_space", obs_space)
    act_space = test_env.action_space
    print("act_space", act_space)

    config["multiagent"] = {
        "policies": {
            "draw": (None, obs_space, act_space, {}),
            "place": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: agent_id.split("_")[0]
    }

    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["log_level"] = "DEBUG"
    config["num_workers"] = 1
    config["rollout_fragment_length"] = 30
    config["train_batch_size"] = 200
    config["horizon"] = 200
    config["no_done_at_end"] = False
    config["framework"] = "torch"
    config["model"] = {
        "custom_model": "pa_model",
    }
    config['n_step'] = 1

    # config["exploration_config"] = {
    #     # The Exploration class to use.
    #     "type": "EpsilonGreedy",
    #     # Config for the Exploration class' constructor:
    #     "initial_epsilon": 0.1,
    #     "final_epsilon": 0.0,
    #     "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
    # }
    config['hiddens'] = []
    config['dueling'] = False
    config['env'] = env_name

    

    tune.run(
        alg_name,
        name="DQN",
        stop={"timesteps_total": 10000000},
        checkpoint_freq=10,
        config=config
        )