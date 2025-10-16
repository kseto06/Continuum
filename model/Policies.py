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
    def __init__(
        self, 
        obs_space: gym.Space, 
        device: Optional[str] = "cpu", 
        solver: Optional[str] = "dopri5", 
        sensitivity: Optional[str] = "adjoint",
    ):
        '''
        Params:
        - obs_dim (int): dimension of the observation space
        - device (str, optional): device to run the model on, defaults to "cpu"
        - solver (str, optional): ODE solver to use, defaults to "dopri5"
        - sensitivity (str, optional): sensitivity method to use (i.e. how gradients for the ODE are backpropagated), defaults to "adjoint"
        '''
        super(Policy, self).__init__()

        obs_dim = np.prod(obs_space.shape)

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
        cnn_model = nn.Sequential(
            DepthCat(1),
            nn.Conv2d(in_channels=obs_dim+1, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv2d(in_channels=32, out_channels=obs_dim, kernel_size=1, stride=1, padding=0),
        )

        self.cnn_model = NeuralODE(
            cnn_model,
            solver=solver,
            sensitivity=sensitivity,
            atol=1e-6,
            rtol=1e-6
        ).to(device)

        # Initialize an MLP-LSTM-NODE architecture. NOTE: this can be changed for different architectures
        lstm_model = nn.Sequential(
            DepthCat(1),
            nn.Linear(obs_dim + 1, 64),
            nn.Tanh(),
            LSTMOutputExtractor(input_size=64, hidden_size=64, num_layers=1, batch_first=True),
            nn.Tanh(),
            nn.Linear(64, obs_dim)
        )
        
        self.lstm_model = NeuralODE(
            lstm_model,
            solver=solver,
            sensitivity=sensitivity,
            atol=1e-6,
            rtol=1e-6
        ).to(device)
    
    def get_mlp_model(self) -> NeuralODE:
        return self.mlp_model
    
    def get_cnn_model(self) -> NeuralODE:
        return self.cnn_model
    
    def get_lstm_model(self) -> NeuralODE:
        return self.lstm_model
    
class LSTMOutputExtractor(nn.Module):
    '''
    Class that extracts the output from an LSTM layer for Neural ODE inputs compatibility 
    '''
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, batch_first: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM output => (output, (h_n, c_n))
        output, _ = self.lstm(x)
        return output

class MlpNodeExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP-NODE architecture as a feature extractor for Stable Baselines3 policies.
    - extends SB3 BaseFeaturesExtractor
    '''
    def __init__(self, obs_space: gym.Space, features_dim: int = 64):
        super(MlpNodeExtractor, self).__init__(obs_space, features_dim)
        obs_dim = np.prod(obs_space.shape)
        self.model = Policy(obs_space=obs_space).get_mlp_model()

        # Projection layer to get the desired feature dimension
        self.head = nn.Sequential(
            nn.Linear(obs_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # returns (ODE solution, t_eval integration interval) from the NODE output
        out = self.model(obs)

        if isinstance(out, tuple):
            _, sol = out
        else:
            sol = out

        # returns 3D tensor (time, batch, features), take the last step for (batch, features)
        if sol.dim() == 3:
            sol = sol[-1]

        return self.head(sol.float())

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
    def __init__(self, obs_space: gym.Space, features_dim: int = 64):
        super(CnnNodeExtractor, self).__init__(obs_space, features_dim)
        obs_dim = np.prod(obs_space.shape)
        self.model = Policy(obs_space=obs_space).get_cnn_model()

        # flatten dimension shape
        with torch.no_grad():
            sample = torch.as_tensor(obs_space.sample()[None]).float()
            if sample.ndim == 3:
                sample = sample.permute(0, 3, 1, 2)  #(1, C, H, W)
                if sample.shape[1] not in (1, 3):
                    sample = sample.permute(0, 3, 1, 2) #(1, H, W, C) -> (1, C, H, W)

            out = self.model(sample)
            B, C, H, W = out.shape

        # linear feature extraction layer
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C * H * W, features_dim),
            nn.ReLU()
        )

        # Projection layer to get the desired feature dimension
        self.head = nn.Sequential(
            nn.Linear(obs_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # returns (ODE solution, t_eval integration interval) from the NODE output
        out = self.feature_extractor(self.model(obs))

        if isinstance(out, tuple):
            _, sol = out
        else:
            sol = out

        # returns 3D tensor (time, batch, features), take the last step for (batch, features)
        if sol.dim() == 3:
            sol = sol[-1]

        return self.head(sol.float())

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
    def __init__(self, obs_space: gym.Space, features_dim: int = 64):
        super(MlpLstmNodeExtractor, self).__init__(obs_space, features_dim)
        obs_dim = np.prod(obs_space.shape)
        self.model = Policy(obs_space=obs_space).get_lstm_model()

        # Projection layer to get the desired feature dimension
        self.head = nn.Sequential(
            nn.Linear(obs_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # returns (ODE solution, t_eval integration interval) from the NODE output
        out = self.model(obs)

        if isinstance(out, tuple):
            _, sol = out
        else:
            sol = out

        # returns 3D tensor (time, batch, features), take the last step for (batch, features)
        if sol.dim() == 3:
            sol = sol[-1]

        return self.head(sol.float())
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim)
        )
