from model.model import train, SB3NodeAgent, MlpNodeExtractor, CnnNodeExtractor, MlpLstmNodeExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, SAC
from torch import nn as nn
from torchdyn.nn import DepthCat
import gymnasium as gym
import mujoco
import sys

'''
Running the code requires the file name and a solver parameter (String) for the input

`python humanoid-walk.py <solver> <total_timesteps> <checkpoint_interval>`
'''

if __name__ == "__main__":
    network_arch = None
    solver = sys.argv[1]
    '''
    This file trains in the Mujoco humanoid env:
    - For training, use render_mode=None to save computational resources
    - For inference, use render_mode=human to visualize the env
    '''

    '''
    VecEnv parameters can be changed in the initialization of the environment.
    NOTE:
    - If training from scratch, initialize a VecNormalize wrapper. For example:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    - If resuming training from a checkpoint, load the VecNormalize wrapper from the saved file. For example:
        env = VecNormalize.load("<vec_normalize_pkl_path>", env)
    '''
    env_name = "Humanoid-v5"
    env = make_vec_env(env_name, n_envs=16, vec_env_cls=SubprocVecEnv)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    env = VecNormalize.load("model/rl-model/humanoid-vectorized/NODE-PPO_rk4_Humanoid-v5_checkpoint_10125312_steps_vecnormalize.pkl", env)

    '''
    # NOTE: 
    # Optionally change the model architecture using nn.Sequential, otherwise it is defaulted in Policies.py. 
    # Example of defining a MLP-NODE architecture below:
    '''

    latent_dim = 64
    features_dim = 512
    network_arch = nn.Sequential(
        DepthCat(1),
        nn.Linear(latent_dim + 1, features_dim),
        nn.Tanh(),
        nn.Linear(features_dim, 256),
        nn.Tanh(),
        nn.Linear(256, latent_dim)
    )

    '''
    NOTE:
    Make sure to change to the corresponding feature extractor class based on defined network architecture above. That is:
    - MlpNodeExtractor for MLP-NODE
    - CnnNodeExtractor for CNN-NODE
    - MlpLstmNodeExtractor for MLP-LSTM-NODE
    '''
    model_arch = MlpNodeExtractor(
        obs_space=env.observation_space,
        features_dim=features_dim,
        latent_dim=latent_dim,
        output_dim=128,
        device="mps",
        solver=solver, # change to euler, rk4, dormand prince, etc. specify in command line argument
        sensitivity="direct",
        network_arch=network_arch
    )

    '''
    NOTE:
    If training from scratch, ensure model_path is set to the desired architecture:
    agent = SB3Agent(env_name=env_name, sb3_class=PPO, model_path=None, model_arch=model_arch)

    If resuming training from a checkpoint, set model_path to the checkpoint file path
    agent = SB3Agent(env_name=env_name, sb3_class=PPO, model_path=f"model/rl-model/{<model name>}", model_arch=None)
    '''
    sb3_class = PPO
    model_path = "model/rl-model/humanoid-vectorized/NODE-PPO_rk4_Humanoid-v5_checkpoint_10125312_steps.zip"
    agent = SB3NodeAgent(env_name=env_name, sb3_class=sb3_class, model_path=model_path, model_arch=model_arch, total_timesteps=int(sys.argv[2]), checkpoint_interval=int(sys.argv[3]))
    train(agent, env, run_name=f"{env_name}-{sb3_class.__name__}-node-{solver}-parallelized")