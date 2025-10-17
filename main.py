from model.model import train, SB3Agent, MlpNodeExtractor, CnnNodeExtractor, MlpLstmNodeExtractor
from model.Policies import LSTMOutputExtractor
from stable_baselines3 import PPO
import gymnasium as gym
from torchdyn.nn import DepthCat
import torch.nn as nn
import numpy as np

if __name__ == "__main__":
    # Possible classic control envs to use are:
    # CartPole-v1, MountainCar-v0/MountainCarContinuous-v0, Acrobot-v1, Pendulum-v1, LunarLander-v2
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    env = gym.make("CartPole-v1", render_mode="human")
    network_arch = None

    '''
    # NOTE: 
    # Optionally change the model architecture using nn.Sequential, otherwise it is defaulted in Policies.py. 
    # Example of defining a MLP-LSTM-NODE architecture below:

    obs_dim = np.prod(env.observation_space.shape)
    network_arch = nn.Sequential(
        DepthCat(1),
        nn.Linear(obs_dim + 1, 64),
        nn.Tanh(),
        LSTMOutputExtractor(input_size=64, hidden_size=64, num_layers=1, batch_first=True),
        nn.Tanh(),
        nn.Linear(64, obs_dim)
    )
    '''

    '''
    NOTE:
    Make sure to change to the corresponding feature extractor class based on defined network architecture above.
    - MlpNodeExtractor for MLP-NODE
    - CnnNodeExtractor for CNN-NODE
    - MlpLstmNodeExtractor for MLP-LSTM-NODE
    '''
    model_arch = MlpNodeExtractor(
        obs_space=env.observation_space,
        features_dim=64,
        device="mps",
        solver="dopri5",
        sensitivity="adjoint",
        network_arch=network_arch
    )

    '''
    NOTE:
    If training from scratch, ensure model_path is set to the desired architecture
    If resuming training from a checkpoint, set model_path to the checkpoint file path
    '''
    agent = SB3Agent(sb3_class=PPO, model_path=None, model_arch=model_arch)
    train(agent, env, run_name="run1")