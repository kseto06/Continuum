import torch 
from torch import nn as nn
import gymnasium as gym
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from torchdyn.nn import DepthCat
from torchdyn.core import NeuralODE

from typing import Optional

class Policy(nn.Module):
    r'''
    Defining the policy network architecture here using PyTorch nn.Module/Sequential
    This class returns a PyTorch model for instantiation in the BaseFeaturesExtractors
    
    NOTE: 
    - The neural network architecture allows us to wrap a NeuralODE where the equation is represented by:

    $$
        y(t_1) = y(t_0) + \int_{t_0}^{t_1} f(y(t), t, \theta) dt
    $$

    - The function f is represented by a neural network outputting dy/dt given input (t, y(t))
    - For the ODE solver, we also use tanh() because for continuous solving we require smooth activation function for stable vector field. 
      ReLU is piecewise with discontinuous derivative so it is not desirable for NODEs.
    '''
    def __init__(self, obs_dim: int, device: Optional[str] = "cpu", solver: Optional[str] = "dopri5", sensitivity: Optional[str] = "adjoint"):
        '''
        Params:
        - obs_dim (int): dimension of the observation space
        - device (str, optional): device to run the model on, defaults to "cpu"
        - solver (str, optional): ODE solver to use, defaults to "dopri5"
        - sensitivity (str, optional): sensitivity method to use (i.e. how gradients for the ODE are backpropagated), defaults to "adjoint"
        '''
        super(Policy, self).__init__()

        # Initialize an MLP-NODE architecture. NOTE: this can be changed for different architectures
        mlp_model = nn.Sequential(
            DepthCat(1), #concats (time, y)
            nn.Linear(obs_dim + 1, 64), #input layer: (t, y_1, ..., y_obs_dim)
            nn.Tanh(), 
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, obs_dim) #output layer: (dy_1/dt, ..., dy_obs_dim/dt)
        )

        self.mlp_model = NeuralODE(
            mlp_model, 
            solver=solver, 
            sensitivity=sensitivity, 
            atol=1e-6, 
            rtol=1e-6
        ).to(device)

        # Initialize a CNN-NODE architecture. NOTE: this can be changed for different architectures
        self.cnn_model = None

        # Initialize an MLP-LSTM-NODE architecture. NOTE: this can be changed for different architectures
        self.lstm_model = None
    
    def get_mlp_model(self) -> NeuralODE:
        return self.mlp_model
    
    def get_cnn_model(self) -> NeuralODE:
        return self.cnn_model
    
    def get_lstm_model(self) -> NeuralODE:
        return self.lstm_model

class MlpNodeExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP-NODE architecture as a feature extractor for Stable Baselines3 policies.
    - extends SB3 BaseFeaturesExtractor
    '''
    def __init__(self, obs_space: gym.Space, features_dim: int = 64):
        super(MlpNodeExtractor, self).__init__(obs_space, features_dim)
        obs_dim = np.prod(obs_space.shape)
        self.model = Policy(obs_dim=obs_dim).get_mlp_model()

        # Projection layer to get the desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(obs_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # return (t_eval, solution) from the NODE output
        out = self.model(obs)

        if isinstance(out, tuple):
            _, sol = out
        else:
            sol = out

        # returns 3D tensor (time, batch, features), take the last step for (batch, features)
        if sol.dim() == 3:
            sol = sol[-1]

        return self.projection(sol.float())

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim)
        )

class CnnNodeExtractor(BaseFeaturesExtractor):
    '''
    Class that defines a CNN-NODE architecture as a feature extractor for Stable Baselines3 policies.
    - extends SB3 BaseFeaturesExtractor 
    '''
    def __init__(self):
        raise NotImplementedError

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim)
        )

class MlpLstmNodeExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP-LSTM-NODE architecture as a feature extractor for Stable Baselines3 policies.
    - extends SB3 BaseFeaturesExtractor 
    '''
    def __init__(self):
        raise NotImplementedError
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim)
        )
