import glob
import os
from typing import Tuple
import json
import ray.tune
from ray import init
from ray.rllib.agents import ppo
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from rlskyjo.environment import skyjo_env
from rlskyjo.game.skyjo import SkyjoGame
from rlskyjo.models.action_mask_model import TorchActionMaskModel, TorchPlayerRelation
from rlskyjo.utils import get_project_root, find_file_in_subdir

torch, nn = try_import_torch()

MAX_CPU = 32

ENV_CONFIG = skyjo_env.DEFAULT_CONFIG.copy()
ENV_CONFIG.update(
    {
        "num_players": 4,
        "observe_other_player_indirect": False,
    }
)

def get_resources():
    """[summary]

    Returns:
        [(int, int)]: num_cpus, num_gpus
    """    
    num_cpus = min(os.cpu_count(), MAX_CPU)
    num_gpus = int(torch.cuda.device_count())
    return num_cpus, num_gpus

def prepare_train(
    prepare_trainer= False,
    env_config=None
    ) -> Tuple[ppo.PPOTrainer, PettingZooEnv, dict]:
    """[summary]

    Returns:
        Tuple[ppo.PPOTrainer, PettingZooEnv, dict]: [description]
    """    
    env_name = "pettingzoo_skyjo"
    if env_config is None:
        env_config = ENV_CONFIG
    # get the Pettingzoo env
    def env_creator(env_config):
        env = skyjo_env.env(env_config=env_config)
        return env

    register_env(env_name, lambda env_config: PettingZooEnv(env_creator(env_config)))
    ModelCatalog.register_custom_model("fc_action_mask_model", TorchActionMaskModel)
    ModelCatalog.register_custom_model("relation_action_mask_model", TorchPlayerRelation)
    # wrap the pettingzoo env in MultiAgent RLLib
    env = PettingZooEnv(env_creator(env_config))
    
    # resources:
    num_cpus, num_gpus = get_resources()
    num_workers = num_cpus - 1
    custom_config = {
        "env": env_name,
        "env_config": env_config,
        "model": {
            "custom_model": "fc_action_mask_model" \
                if env_config["observe_other_player_indirect"] else "relation_action_mask_model",
                # use model fitting to the action space
        },
        "framework": "torch",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        # "train_batch_size": 4000,
        "multiagent": {
            "policies": {
                name: (None, env.observation_space, env.action_space, {})
                for name in env.agents
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
    }

    # get trainer

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(custom_config)

    if prepare_trainer:
        trainer = ppo.PPOTrainer(config=ppo_config)
    else:
        trainer = None
    return trainer, env, ppo_config


def steps_train(trainer, max_steps=2e6):
    # run manual training loop and print results after each iteration
    iters = 0
    while True:
        iters += 1
        result = trainer.train()
        print(pretty_print(result))
        # stop training if the target train steps or reward are reached
        if result["timesteps_total"] >= max_steps:
            print(
                "training done, because max_steps"
                f"{max_steps} {result['timesteps_total']} reached"
            )
            break
    # manual test loop
    return trainer


def train_ray(ppo_config: dict, seconds_max: int = 120, restore: str = None):
    """[summary]

    Args:
        ppo_config (dict): config
        seconds_max (int, optional): timesteps to train. Defaults to 120.
        restore (str, optional): path_to_trained_agent_checkpoint. Defaults to None.

    Returns:
        analysis: [description]
    """    
    extra_kwargs = {}
    if restore is not None:
        extra_kwargs.update({"restore": restore})
    
    
    
    analysis = ray.tune.run(
        ppo.PPOTrainer,
        config=ppo_config,
        local_dir=os.path.join(get_project_root(), "models"),
        time_budget_s=seconds_max,
        checkpoint_at_end=True,
        checkpoint_freq=128 if seconds_max > 3600 else 1,
        **extra_kwargs
    )
    return analysis


def load_ray(path, ppo_config):
    """
    Load a trained RLlib agent from the specified path.
    Call this before testing a trained agent.
    :param path:
        Path pointing to the agent's saved checkpoint (only used for RLlib agents)
    :param ppo_config:
        dict config
    """
    trainer = ppo.PPOTrainer(config=ppo_config)
    trainer.restore(path)
    return trainer


def sample_trainer(trainer, env):
    print("Finished training. Running manual test/inference loop.")
    obs = env.reset()
    done = {"__all__": False}
    # run one iteration until done
    env.render()
    for i in range(10000):
        if done["__all__"]:
            print("game done")
            break
        # get agent from current observation
        agent = list(obs.keys())[0]

        # format observation dict
        # print(obs)
        obs = obs[agent]
        for i in obs:
            obs[i] = obs[i].astype(float)
        # env.render()

        # get deterministic action
        # trainer.compute_single_action(obs, policy_id=agent)
        policy = trainer.get_policy(policy_id=agent)
        action_exploration_policy, _, action_info = policy.compute_single_action(obs)
        logits = action_info["action_dist_inputs"]
        action = logits.argmax()
        action = action_exploration_policy
        # print("agent ", agent, " action ", SkyjoGame.render_action_explainer(action))
        obs, reward, done, infos = env.step({agent: action})
        # observations contain original observations and the action mask
        # print(f"Obs: {obs}, Action: {action}, done: {done}")

    env.render()
    print(env.env.rewards)


def tune_training_loop(seconds_max: int = 120):
    """train trainer and sample"""
    trainer, env, ppo_config = prepare_train()

    # train trainer
    analysis = train_ray(ppo_config, 
                         seconds_max=seconds_max)
    # reload the checkpoint
    last_chpt_path = analysis._checkpoints[-1]["local_dir"]
    checkpoint_file = find_file_in_subdir(last_chpt_path, "checkpoint-*", regex_match=".*\/checkpoint-[0-9]{0,9}$")

    trainer_trained = load_ray(checkpoint_file, ppo_config)

    # sample trainer
    sample_trainer(trainer_trained, env)
    return last_chpt_path

def continual_train(restore_path: str, seconds_max: int = 120,):
    """continoue to train trainer and sample"""
    _, env, ppo_config = prepare_train()
    # train trainer
    analysis = train_ray(ppo_config, 
                         seconds_max=seconds_max,
                         restore=find_file_in_subdir(restore_path, "checkpoint-*", regex_match=".*\/checkpoint-[0-9]{0,9}$"))
    # reload the checkpoint
    last_chpt_path = analysis._checkpoints[-1]["local_dir"]
    checkpoint_file = find_file_in_subdir(last_chpt_path, "checkpoint-*", regex_match=".*\/checkpoint-[0-9]{0,9}$")
    trainer_trained = load_ray(checkpoint_file, ppo_config)
    sample_trainer(trainer_trained, env)

def manual_training_loop(timesteps_total=10000):
    """train trainer and sample"""

    trainer, env, ppo_config = prepare_train(prepare_trainer=True, config=skyjo_env.DEFAULT_CONFIG.copy())
    trainer_trained = steps_train(trainer, max_steps=timesteps_total)

    sample_trainer(trainer_trained, env)

def init_ray(local=False):
    num_cpus, num_gpus = get_resources()
    
    if local:
        init(local_mode=True)
    else:
        init(num_cpus=None, num_gpus=num_gpus)

if __name__ == "__main__":
    init_ray(True)
    last_chpt_path = tune_training_loop(60*60*23) # train for 1 min
    # continual_train(last_chpt_path, 60 // 2) # load model and train for 30 seconds
