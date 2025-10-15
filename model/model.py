from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, Type

import torch
from torch import nn as nn

import numpy as np
from matplotlib import pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO, A2C, DDPG, DQN, TD3, SAC
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.results_plotter import ts2xy, load_results

from .Policies import MlpNodeExtractor, CnnNodeExtractor, MlpLstmNodeExtractor

class Agent(ABC):
    def __init__(self, 
            model_path: str, 
            total_timesteps: int = 100_000,
            n_steps: int = 2048,
            batch_size: int = 128,
            num_envs: int = 8,
            checkpoint_interval: int = 20_000,
            device: Optional[str] = "cpu"
        ):
        '''
        Abstract base class for defining an agent to be trained in a given environment.
        params:
        - model_path (str): path to the pre-trained model, if None a new model will be initialized
        - total_timesteps (int): total number of timesteps to train the model for, defaults to 100,000
        - n_steps (int): number of steps to run for each environment per update, defaults to 2048
        - batch_size (int): minibatch size, defaults to 128
        - num_envs (int): number of parallel environments, defaults to 8
        - checkpoint_interval (int): frequency (in timesteps) to save the model, defaults to 20,000
        - device (str, optional): device to run the model on, defaults to "cpu"
        '''
        # if model_path is None:
        #     raise Exception("model_path is required")

        self.model_path: Optional[str] = model_path
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.checkpoint_interval = checkpoint_interval
        self.device = device
    
    def get_env_info(self, env: gym.Env):
        if isinstance(env, Monitor):
            self.env = env.env
        else:
            self.env = env
        
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.initialize()
    
    @abstractmethod
    def predict(self, obs: gym.spaces.Box):
        pass

    @abstractmethod
    def set_checkpoint(self, save_frequency: int) -> CheckpointCallback:
        return

    def reset(self) -> None:
        return
    
    def initialize(self) -> None:
        pass

    def learn(self, log_interval: int, verbose: int) -> None:
        pass
    
class SB3Agent(Agent):
    '''
    Defines the class for initializing a Stable Baselines3 model for training and inference.
    '''
    def __init__(
            self, 
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            model_path: str = None,
            model_arch: BaseFeaturesExtractor = None,
        ):
        '''
        params:
        - sb3_class (Type[BaseAlgorithm], optional): Stable Baselines3 class to use (e.g PPO, A2C, DDPG, DQN, TD3, SAC), defaults to PPO
        - model_path (str): path to the pre-trained model, if None a new model will be initialized
        - model_arch (BaseFeaturesExtractor): feature extractor architecture to use (e.g. MlpNodeExtractor, CnnNodeExtractor, MlpLstmNodeExtractor)
        '''
        self.sb3_class = sb3_class
        super().__init__(model_path)

        self.model_arch = model_arch
        if isinstance(model_arch, MlpNodeExtractor):
            self.model_policy = "MlpPolicy"
        elif isinstance(model_arch, CnnNodeExtractor):
            self.model_policy = "CnnPolicy"
        elif isinstance(model_arch, MlpLstmNodeExtractor):
            self.model_policy = "MlpLstmPolicy"

    def initialize(self) -> None:
        '''
        Void function to initialize the SB3 model, either loading a pre-trained model or initializing a new one
        '''
        if self.model_path is not None:
            self.model = self.sb3_class.load(self.model_path)
        else:
            self.model = self.sb3_class(
                self.model_policy,
                self.env,
                policy_kwargs=self.model_arch.get_policy_kwargs(),
                verbose=1,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                ent_coef=0.01
            )

    def predict(self, obs: gym.spaces.Box) -> np.ndarray:
        '''
        Function to predict the action to take given an observation
        params:
        - obs (gym.spaces.Box): observation from the environment

        returns:
        - action (np.ndarray): action to take
        '''
        action, _states = self.model.predict(obs, deterministic=True)
        return action

    def set_checkpoint(self, save_frequency: int) -> CheckpointCallback:
        '''
        Function to set up a checkpoint callback to save the model at regular intervals
        params:
        - save_frequency (int): frequency (in timesteps) to save the model
        returns:
        - checkpoint_callback (CheckpointCallback): checkpoint callback to pass to the SB3 model's learn
        '''
        checkpoint_callback = CheckpointCallback(
            save_freq=save_frequency, 
            save_path=f"./rl-model/",
            name_prefix=f"{self.sb3_class.__name__}_{self.env.spec.id}_checkpoint"
        )

        return checkpoint_callback
    
    def learn(self, log_interval: int = 1, verbose: int = 1) -> None:
        '''
        Function to train the SB3 model
        params:
        - log_interval (int): interval to log the training progress
        - verbose (int): verbosity level
        '''
        self.model.set_env(self.env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=self.total_timesteps, 
            log_interval=log_interval,
            progress_bar=True,
            callback=self.set_checkpoint(save_frequency=self.checkpoint_interval)
        )

class RecurrentPPOAgent(Agent):
    '''
    Defines the class for initializing a RecurrentPPO model (i.e. a PPO model with memory) for training and inference.
    '''
    def __init__(self, model_path: str, policy_kwargs: Dict[str, Any]):
        '''
        params:
        - model_path (str): path to the pre-trained model, if None a new model will be initialized
        - policy_kwargs (Dict[str, Any]): dictionary of keyword arguments to pass to the policy
        '''
        super.__init__(model_path)
        self.lstm_states = None
        if policy_kwargs is None and not(set(["activation_fn", "lstm_hidden_size", "net_arch", "shared_lstm", "enable_critic_lstm", "share_features_extractor"]) - set(policy_kwargs.keys())):
            print("ERROR: Invalid policy_kwargs, defaulting to base policy_kwargs")
            policy_kwargs = {
                "activation_fn": nn.ReLU,
                "lstm_hidden_size": 256, 
                "net_arch": [dict(pi=[32, 32], vf=[32, 32])],
                "shared_lstm": True,
                "enable_critic_lstm": False,
                "share_features_extractor": True,
            }

def plot_results(log_folder: str, sb3_class_name: str, env_name: str, title: str = "Learning Curve") -> None:
    '''
    Function to plot the results of training (rewards vs. timesteps)

    params:
    - log_folder (str): folder where the results are stored
    - sb3_class_name (str): name of the SB3 class used (e.g PPO, A2C, DDPG, DQN, TD3, SAC)
    - env_name (str): name of the environment used (e.g CartPole-v1)
    - title (str): title of the plot
    '''
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    
    weights = np.repeat(1.0, 50) / 50
    y = np.convolve(y, weights, "valid")
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")

    plt.savefig(f"{log_folder}/{sb3_class_name}-{env_name}-learning_curve.png")


def train(agent: Agent, env: gym.Env):
    '''
    Function to train a general agent in a given env
    params:
    - agent (Agent): agent to be trained
    - env (gym.Env): environment to train the agent in
    '''
    agent.get_env_info(env)
    agent.learn(log_interval=1, verbose=1)
    plot_results(f"./rl-model/", agent.sb3_class.__name__, env.spec.id, title=f"{agent.sb3_class.__name__} on {env.spec.id}")

    env.close()