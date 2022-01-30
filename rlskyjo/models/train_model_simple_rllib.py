import os
from typing import Tuple

from ray import init
from ray.rllib.agents import ppo
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from rlskyjo.environment import skyjo_env
from rlskyjo.game.skyjo import SkyjoGame
from rlskyjo.models.action_mask_model import TorchActionMaskModel

torch, nn = try_import_torch()
init(num_cpus=os.cpu_count())


def prepare_train() -> Tuple[ppo.PPOTrainer, PettingZooEnv]:
    env_name = "pettingzoo_skyjo"

    # get the Pettingzoo env
    def env_creator():
        env = skyjo_env.env(**skyjo_env.DEFAULT_CONFIG)
        return env

    register_env(env_name, lambda config: PettingZooEnv(env_creator()))
    ModelCatalog.register_custom_model("pa_model2", TorchActionMaskModel)
    # wrap the pettingzoo env in MultiAgent RLLib
    env = PettingZooEnv(env_creator())
    custom_config = {
        "env": env_name,
        "model": {
            "custom_model": "pa_model2",
        },
        "framework": "torch",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(torch.cuda.device_count()),
        "num_workers": os.cpu_count() - 1,
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

    trainer = ppo.PPOTrainer(config=ppo_config)

    return trainer, env


def train(trainer, max_steps=2e6, max_iters=100):
    # run manual training loop and print results after each iteration
    max_steps = 2e6
    max_iters = 100
    for iters in range(max_iters):
        result = trainer.train()
        if iters % 2 == 0:
            print(pretty_print(result))
        # stop training if the target train steps or reward are reached
        if result["timesteps_total"] >= max_steps:
            print(
                f"training done, because max_steps {max_steps} {result['timesteps_total']} reached"
            )
            break
    else:
        print(f"training done, because max_iters {max_iters} reached")
    # manual test loop
    print("Finished training. Running manual test/inference loop.")
    return trainer


def sample_trainer(trainer, env):
    obs = env.reset()
    done = {"__all__": False}
    # run one iteration until done

    for i in range(10000):
        if done["__all__"]:
            print("game done")
            break
        # get agent from current observation
        agent = list(obs.keys())[0]

        # format observation dict
        print(obs)
        obs = obs[agent]
        env.render()

        # get deterministic action
        # trainer.compute_single_action(obs, policy_id=agent)
        policy = trainer.get_policy(policy_id=agent)
        action_exploration_policy, _, action_info = policy.compute_single_action(obs)
        logits = action_info["action_dist_inputs"]
        action = logits.argmax()
        print("agent ", agent, " action ", SkyjoGame.render_action_explainer(action))
        obs, reward, done, _ = env.step({agent: action})
        # observations contain original observations and the action mask
        # print(f"Obs: {obs}, Action: {action}, done: {done}")

    env.render()
    print(env.env.rewards)


if __name__ == "__main__":
    trainer, env = prepare_train()
    trainer_trained = train(trainer)
    sample_trainer(trainer_trained, env)
