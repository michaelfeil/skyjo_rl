from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
import supersuit as ss
from rlskyjo.environment import simple_skyjo_env
from pettingzoo.utils.conversions import to_parallel

def env_creator():
    env_name = "rlskyjo_baselines"
    env = simple_skyjo_env.env(**{"name":env_name, "num_players": 2})
    env = to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')
    return env

env = env_creator()
model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
model.learn(total_timesteps=100)
model.save("policy")

