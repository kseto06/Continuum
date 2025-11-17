import os
from abc import ABC, abstractmethod
from typing import Optional, Union, Type, Dict

import torch
from torch import nn as nn

import numpy as np
from matplotlib import pyplot as plt

import gymnasium as gym
import pygame

from stable_baselines3 import PPO, A2C, DDPG, DQN, TD3, SAC
from sb3_contrib import RecurrentPPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .Policies import MlpNodeExtractor, CnnNodeExtractor, MlpLstmNodeExtractor

import subprocess as sp
import atexit

class CheckpointCallback(BaseCallback):
    '''
    Class to define a custom checkpoint callback for saving the model at regular intervals
    Allows for compatibility across SubprocVecEnvs 
    params:
    - save_frequency (int): frequency (in timesteps) to save the model
    - save_path (str): path to save the model
    - verbose (int): verbosity level
    '''
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_frequency = save_freq 
        self.save_path = save_path
        self.name_prefix = name_prefix

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_frequency == 0:
            file_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}.zip"
            )
            self.model.save(file_path)

        return True

class Agent(ABC):
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
    def __init__(self, 
            model_path: str, 
            total_timesteps: int = 100_000,
            n_steps: int = 2048,
            batch_size: int = 4096,
            num_envs: int = 8,
            env_name: str = None,
            checkpoint_interval: int = 20_000,
            device: Optional[str] = "cpu"
        ):
        if env_name is None:
            raise Exception("Input a valid Gym environment name for initialization.")

        self.model_path: Optional[str] = model_path
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.checkpoint_interval = checkpoint_interval
        self.device = device   
        self.env_name = env_name

        atexit.register(self.save)
    
    def get_env_info(self, env: Union[gym.Env | DummyVecEnv | SubprocVecEnv]):
        if isinstance(env, Monitor):
            self.env = env.env
        else:
            self.env = env

        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.initialize()
        self.env_id = self.get_env_id()

    def get_env_id(self) -> str:
        return self.env_name
    
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

    def learn(self, log_interval: int, verbose: int, run_name: str) -> None:
        pass

    # Saving model on exit
    def save(self, model: Union[BaseAlgorithm | RecurrentPPO], file_name: str) -> None:
        '''
        Void function to save the model to the specified path on system process exit
        '''
        print("\n System process exited, saving model at current timestep...")
        model.save(f"./model/rl-model/{file_name}.zip")

class SB3NodeAgent(Agent):
    '''
    Defines the class for initializing a Neural ODE Stable Baselines3 model for training and inference.
    params:
    - sb3_class (Type[BaseAlgorithm], optional): Stable Baselines3 class to use (e.g PPO, A2C, DDPG, DQN, TD3, SAC), defaults to PPO
    - model_path (str): path to the pre-trained model, if None a new model will be initialized
    - model_arch (BaseFeaturesExtractor): feature extractor architecture to use (e.g. MlpNodeExtractor, CnnNodeExtractor, MlpLstmNodeExtractor)
    - total_timesteps (int): total number of timesteps to train the model for, defaults to 100,000
    - n_steps (int): number of steps to run for each environment per update, defaults to 2048
    - batch_size (int): minibatch size, defaults to 128
    - num_envs (int): number of parallel environments, defaults to 8
    - checkpoint_interval (int): frequency (in timesteps) to save the model, defaults to 20,000
    '''
    def __init__(
            self, 
            env_name: str = None,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            model_path: str = None,
            model_arch: BaseFeaturesExtractor = None,
            total_timesteps: int = 100_000,
            n_steps: int = 2048,
            batch_size: int = 4096,
            checkpoint_interval: int = 20_000
        ):
        self.sb3_class = sb3_class
        super().__init__(model_path, total_timesteps, env_name=env_name, n_steps=n_steps, batch_size=batch_size, checkpoint_interval=checkpoint_interval)

        self.model_arch = model_arch

        if self.model_arch is None and self.model_path is None:
            raise Exception("ERROR: model_arch or model_path is not provided for starting or continuing a new training session, please provide at least one of these details.")

        if isinstance(model_arch, MlpNodeExtractor):
            self.model_policy = "MlpPolicy"
        elif isinstance(model_arch, CnnNodeExtractor):
            self.model_policy = "CnnPolicy"
        elif isinstance(model_arch, MlpLstmNodeExtractor):
            self.model_policy = "MlpLstmPolicy"
            if self.sb3_class != RecurrentPPO:
                print("WARNING: MlpLstmNodeExtractor is only compatible with RecurrentPPO, defaulting to RecurrentPPO")
                self.sb3_class = RecurrentPPO

    def initialize(self) -> None:
        '''
        Void function to initialize the SB3 model, either loading a pre-trained model or initializing a new one
        '''
        if self.model_path is not None:
            #Loading in base model path
            try:
                self.model = self.sb3_class.load(self.model_path)
            #Loading in pretrained model trained on one env for SubprocEnv
            except AssertionError:
                self.model = self.sb3_class.load(self.model_path, env=self.env)

            print(f"Resuming training from {self.model.num_timesteps:,} steps...")
        else:
            self.model = self.sb3_class(
                self.model_policy,
                self.env,
                policy_kwargs=self.model_arch.get_policy_kwargs(),
                verbose=1,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                ent_coef=0.01,
                tensorboard_log="./model/rl-model/logs/",
            )
            print("Initializing training from scratch...")

    def predict(self, obs: gym.spaces.Box) -> np.ndarray:
        '''
        Function to predict the action to take given an observation
        params:
        - obs (gym.spaces.Box): observation from the environment

        returns:
        - action (np.ndarray): action to take
        '''
        action, _states = self.model.predict(obs)
        return action

    def set_checkpoint(self, save_frequency: int = 10000) -> CheckpointCallback:
        '''
        Function to set up a checkpoint callback to save the model at regular intervals
        params:
        - save_frequency (int): frequency (in timesteps) to save the model
        
        returns:
        - checkpoint_callback (CheckpointCallback): checkpoint callback to pass to the SB3 model's learn
        '''
        checkpoint_callback = CheckpointCallback(
            save_freq=save_frequency, 
            save_path=f"./model/rl-model/",
            name_prefix=f"NODE-{self.sb3_class.__name__}_{self.model_arch.solver}_{self.env_id}_checkpoint"
        )

        return checkpoint_callback
    
    def set_evaluation(self) -> EvalCallback:
        '''
        Function to set up an evaluation callback to evaluate the model at regular intervals
        returns:
        - eval_callback (EvalCallback): evaluation callback to pass to the SB3 model's learn
        '''
        eval_env = VecNormalize(
            DummyVecEnv([lambda: gym.make(self.env_id)]), 
            norm_obs=True, 
            norm_reward=False, 
            training=False,
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./model/rl-model/",
            log_path=f"./model/rl-model/",
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )

        return eval_callback
    
    def learn(self, log_interval: int = 1, verbose: int = 1, run_name: str = "run1") -> None:
        '''
        Void function to train the SB3 model and open Tensorboard logs 

        params:
        - log_interval (int): interval to log the training progress
        - verbose (int): verbosity level
        '''
        self.model.set_env(self.env)
        self.model.verbose = verbose   

        sp.Popen(
            ["tensorboard", "--logdir", "./model/rl-model/logs/", "--port", "6006"], 
            stdout=sp.DEVNULL, 
            stderr=sp.STDOUT
        )

        try:
            self.model.learn(
                total_timesteps=self.total_timesteps, 
                log_interval=log_interval,
                progress_bar=True,
                callback=[self.set_evaluation(), self.set_checkpoint(save_frequency=self.checkpoint_interval)],
                reset_num_timesteps=False,
                tb_log_name=f"NODE-{self.sb3_class.__name__}_{self.env_id}_{run_name}"
            )
        except KeyboardInterrupt:
            self.save(
                model=self.model, 
                file_name=f"NODE-{self.sb3_class.__name__}_{self.model_arch.solver}_{self.env_id}_checkpoint_{self.model.num_timesteps}_steps"
            )
            print(f"[INTERRUPT] Model saved at: model/rl-model/NODE-{self.sb3_class.__name__}_{self.env_id}_checkpoint_{self.model.num_timesteps}_steps")

class SB3Agent(Agent):
    '''
    Defines a class for training a basic agent using a base Stable-Baselines3 algorithm for benchmarking:

    params:
    - sb3_class (Type[BaseAlgorithm], optional): Stable Baselines3 class to use (e.g PPO, A2C, DDPG, DQN, TD3, SAC), defaults to PPO
    - model_path (str): path to the pre-trained model, if None a new model will be initialized
    - model_arch (BaseFeaturesExtractor): feature extractor architecture to use (e.g. MlpNodeExtractor, CnnNodeExtractor, MlpLstmNodeExtractor)
    - total_timesteps (int): total number of timesteps to train the model for, defaults to 100,000
    - n_steps (int): number of steps to run for each environment per update, defaults
    - batch_size (int): minibatch size, defaults to 128
    - num_envs (int): number of parallel environments, defaults to 8
    - checkpoint_interval (int): frequency (in timesteps) to save the model, defaults to 20,000
    ''' 
    def __init__(
        self, 
        env_name: str = None,
        sb3_class: Optional[Type[BaseAlgorithm] | RecurrentPPO] = PPO,
        model_path: str = None,
        model_policy: Optional[str] = "MlpPolicy",
        policy_kwargs: Optional[Dict] = None, 
        n_steps: int = 2048,
        batch_size: int = 4096,
        total_timesteps: int = 100_000,
        checkpoint_interval: int = 20_000
    ):
        super().__init__(model_path, total_timesteps, env_name=env_name, n_steps=n_steps, batch_size=batch_size, checkpoint_interval=checkpoint_interval)

        self.sb3_class = sb3_class
        self.model_policy = model_policy
        if policy_kwargs is None:
            self.policy_kwargs = None
        else:
            self.policy_kwargs = policy_kwargs

    def initialize(self) -> None:
        '''
        Void function to initialize the SB3 model, either loading a pre-trained model or initializing a new one
        '''
        if self.model_path is not None:
            #Loading in base model path
            try:
                self.model = self.sb3_class.load(self.model_path)
            #Loading in pretrained model trained on one env for SubprocEnv
            except AssertionError:
                self.model = self.sb3_class.load(self.model_path, env=self.env)
            print(f"Resuming training from {self.model.num_timesteps:,} steps...")
        else:
            self.model = self.sb3_class(
                self.model_policy,
                self.env,
                policy_kwargs=self.policy_kwargs,
                verbose=1,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                ent_coef=0.01,
                tensorboard_log="./model/rl-model/logs/",
            )
            print("Initializing training from scratch...")

    def predict(self, obs: gym.spaces.Box) -> np.ndarray:
        '''
        Function to predict the action to take given an observation
        params:
        - obs (gym.spaces.Box): observation from the environment

        returns:
        - action (np.ndarray): action to take
        '''
        action, _states = self.model.predict(obs)
        return action

    def set_checkpoint(self, save_frequency: int = 10000) -> CheckpointCallback:
        '''
        Function to set up a checkpoint callback to save the model at regular intervals
        params:
        - save_frequency (int): frequency (in timesteps) to save the model
        
        returns:
        - checkpoint_callback (CheckpointCallback): checkpoint callback to pass to the SB3 model's learn
        '''
        checkpoint_callback = CheckpointCallback(
            save_freq=save_frequency, 
            save_path=f"./model/rl-model/",
            name_prefix=f"{self.sb3_class.__name__}_{self.env_id}_checkpoint"
        )

        return checkpoint_callback
    
    def learn(self, log_interval: int = 1, verbose: int = 1, run_name: str = "run1") -> None:
        '''
        Void function to train the SB3 model and open Tensorboard logs 

        params:
        - log_interval (int): interval to log the training progress
        - verbose (int): verbosity level
        '''
        self.model.set_env(self.env)
        self.model.verbose = verbose   

        sp.Popen(
            ["tensorboard", "--logdir", "./model/rl-model/logs/", "--port", "6006"], 
            stdout=sp.DEVNULL, 
            stderr=sp.STDOUT
        )

        try:
            self.model.learn(
                total_timesteps=self.total_timesteps, 
                log_interval=log_interval,
                progress_bar=True,
                callback=self.set_checkpoint(save_frequency=self.checkpoint_interval),
                reset_num_timesteps=False,
                tb_log_name=f"{self.sb3_class.__name__}_{self.env_id}_{run_name}"
            )
        except KeyboardInterrupt:
            self.save(
                model=self.model, 
                file_name=f"{self.sb3_class.__name__}_{self.env_id}_checkpoint_{self.model.num_timesteps}_steps"
            )
            print(f"[INTERRUPT] Model saved at: model/rl-model/{self.sb3_class.__name__}_{self.env_id}_checkpoint_{self.model.num_timesteps}_steps")


def plot_results(log_folder: str, sb3_class_name: str, solver_name: str, env_name: str, title: str = "Learning Curve") -> None:
    '''
    Function to plot the results of training (rewards vs. timesteps), saves the PNG plot in the log_folder

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

    plt.savefig(f"{log_folder}/NODE-{sb3_class_name}_{solver_name}_{env_name}_learning-curve.png")

def train(agent: Agent, env: gym.Env, run_name: str = "run0") -> None:
    '''
    Function to train a general agent (SB3NodeAgent or SB3Agent) in a given env
    params:
    - agent (Agent): agent to be trained
    - env (gym.Env): environment to train the agent in
    - run_name (str): name of the training run
    '''
    env.reset()
    agent.get_env_info(env)
    env_id = agent.env_id
    pygame.display.set_caption(f"{env_id}_{getattr(agent, 'model_policy', 'Unknown')}_{agent.sb3_class.__name__}")
    
    agent.learn(log_interval=1, verbose=1, run_name=run_name)

    model_arch = getattr(agent, "model_arch", None)
    if model_arch is not None and isinstance(model_arch, BaseFeaturesExtractor):
        solver_name = getattr(model_arch, "solver", type(model_arch).__name__)
    else:
        solver_name = "default"

    plot_results(f"./model/rl-model/", agent.sb3_class.__name__, solver_name, env_id, title=f"{agent.sb3_class.__name__} on {env_id}")

    env.close()