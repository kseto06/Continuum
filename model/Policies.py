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
        latent_dim: int = 64,
        features_dim: int = 64,
        mlp_model: Optional[nn.Sequential] = None,
        cnn_model: Optional[nn.Sequential] = None,
        lstm_model: Optional[nn.Sequential] = None
    ):
        '''
        Params:
        - obs_dim (int): dimension of the observation space
        - device (str, optional): device to run the model on, defaults to "cpu"
        - solver (str, optional): ODE solver to use, defaults to "dopri5".
            - The list of available solvers are:
                - "dopri5", "adams", "euler", "midpoint", "rk4", "explicit_adams", "implicit_adams", "bdf"
        - sensitivity (str, optional): sensitivity method to use (i.e. how gradients for the ODE are backpropagated), defaults to "adjoint"
        '''
        super(Policy, self).__init__()

        obs_dim = np.prod(obs_space.shape)

        # Initialize an MLP-NODE architecture. NOTE: this can be changed for different architectures
        if mlp_model is None:
            print("No MLP model provided, using default architecture")
            mlp_model = nn.Sequential(
                DepthCat(1), #concats (time, y)
                nn.Linear(latent_dim + 1, features_dim), #input layer: (t, y_1, ..., y_obs_dim)
                nn.Tanh(), 
                nn.Linear(features_dim, features_dim),
                nn.Tanh(),
                nn.Linear(features_dim, latent_dim) #output layer: (dy_1/dt, ..., dy_obs_dim/dt)
            )

        self.mlp_model = NeuralODE(
            mlp_model, 
            solver=solver, 
            sensitivity=sensitivity, 
            atol=1e-6, 
            rtol=1e-6
        ).to(device)

        # Initialize a CNN-NODE architecture. NOTE: this can be changed for different architectures
        if cnn_model is None:
            print("No CNN model provided, using default architecture")
            cnn_model = nn.Sequential(
                DepthCat(1),
                nn.Conv2d(in_channels=latent_dim+1, out_channels=32, kernel_size=3, stride=1, padding=0),
                nn.Tanh(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
                nn.Tanh(),
                nn.Conv2d(in_channels=32, out_channels=latent_dim, kernel_size=1, stride=1, padding=0),
            )

        self.cnn_model = NeuralODE(
            cnn_model,
            solver=solver,
            sensitivity=sensitivity,
            atol=1e-6,
            rtol=1e-6
        ).to(device)

        # Initialize an MLP-LSTM-NODE architecture. NOTE: this can be changed for different architectures
        if lstm_model is None:
            print("No MLP-LSTM model provided, using default architecture")
            lstm_model = nn.Sequential(
                DepthCat(1),
                nn.Linear(latent_dim + 1, features_dim),
                nn.Tanh(),
                LSTMOutputExtractor(input_size=features_dim, hidden_size=features_dim, num_layers=1, batch_first=True),
                nn.Tanh(),
                nn.Linear(features_dim, latent_dim)
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

class MlpNodeExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP-NODE architecture as a feature extractor for Stable Baselines3 policies.
    - extends SB3 BaseFeaturesExtractor
    '''
    def __init__(
        self, 
        obs_space: gym.Space, 
        features_dim: int = 64,
        latent_dim: int = 64,
        output_dim: int = 32, 
        device: Optional[str] = "cpu", 
        solver: Optional[str] = "dopri5", 
        sensitivity: Optional[str] = "adjoint",
        network_arch: Optional[nn.Sequential] = None,
    ):
        '''
        Params:
        - obs_space (gym.Space): observation space of the environment
        - features_dim (int): dimension of the features extracted
        - device (str, optional): device to run the model on, defaults to "cpu"
        - solver (str, optional): ODE solver to use, defaults to "dopri5"
        - sensitivity (str, optional): sensitivity method to use (i.e. how gradients for the ODE are backpropagated), defaults to "adjoint"
        - network_arch (nn.Sequential, optional): custom MLP architecture to use for the NODE, defaults to default network architecture in Policy class
        '''
        super(MlpNodeExtractor, self).__init__(obs_space, output_dim)
        #super(MlpNodeExtractor, self).__init__(obs_space, features_dim)
        obs_dim = np.prod(obs_space.shape)

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.Tanh()
        )

        self.model = Policy(
            obs_space=obs_space, 
            device=device, 
            solver=solver, 
            sensitivity=sensitivity, 
            latent_dim=latent_dim,
            features_dim=features_dim,
            mlp_model=network_arch
        ).get_mlp_model()

        # Projection layer to get the desired feature dimension
        self.head = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            #nn.Linear(obs_dim, features_dim),
            nn.ReLU()
        )

        self.solver = solver

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        '''
        Params:
        - obs (torch.Tensor): input observation tensor

        Returns:
        - features (torch.Tensor): extracted feature tensor
        '''
        out = self.encoder(obs)
        # returns (ODE solution, t_eval integration interval) from the NODE output
        out = self.model(out)

        if isinstance(out, tuple):
            _, sol = out
        else:
            sol = out

        # returns 3D tensor (time, batch, features), take the last step for (batch, features)
        if sol.dim() == 3:
            sol = sol[-1]

        return self.head(sol.float())

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, output_dim: int = 32, latent_dim: int = 64) -> dict:
        '''
        Params:
        - features_dim (int): dimension of the features extracted

        Returns:
        - the policy kwargs (Dict) for Stable Baselines3 model initialization
        '''
        return dict(
            features_extractor_class=cls,
            #features_extractor_kwargs=dict(features_dim=features_dim)
            features_extractor_kwargs=dict(features_dim=features_dim, output_dim=output_dim, latent_dim=latent_dim)
        )

class CnnNodeExtractor(BaseFeaturesExtractor):
    '''
    Class that defines a CNN-NODE architecture as a feature extractor for Stable Baselines3 policies.
    - extends SB3 BaseFeaturesExtractor 
    '''
    def __init__(
        self, 
        obs_space: gym.Space, 
        features_dim: int = 64, 
        device: Optional[str] = "cpu", 
        solver: Optional[str] = "dopri5", 
        sensitivity: Optional[str] = "adjoint",
        network_arch: Optional[nn.Sequential] = None,
    ):
        '''
        Params:
        - obs_space (gym.Space): observation space of the environment
        - features_dim (int): dimension of the features extracted
        - device (str, optional): device to run the model on, defaults to "cpu"
        - solver (str, optional): ODE solver to use, defaults to "dopri5"
        - sensitivity (str, optional): sensitivity method to use (i.e. how gradients for the ODE are backpropagated), defaults to "adjoint"
        - network_arch (nn.Sequential, optional): custom MLP architecture to use for the NODE, defaults to default network architecture in Policy class
        '''
        super(CnnNodeExtractor, self).__init__(obs_space, features_dim)
        obs_dim = np.prod(obs_space.shape)
        self.model = Policy(
            obs_space=obs_space,
            device=device,
            solver=solver,
            sensitivity=sensitivity,
            cnn_model=network_arch
        ).get_cnn_model()

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

        self.solver = solver

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        '''
        Params:
        - obs (torch.Tensor): input observation tensor

        Returns:
        - features (torch.Tensor): extracted feature tensor
        '''
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
        '''
        Params:
        - features_dim (int): dimension of the features extracted

        Returns:
        - the policy kwargs (Dict) for Stable Baselines3 model initialization
        '''
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim)
        )

class MlpLstmNodeExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP-LSTM-NODE architecture as a feature extractor for Stable Baselines3 policies.
    - extends SB3 BaseFeaturesExtractor 
    '''
    def __init__(
        self, 
        obs_space: gym.Space, 
        features_dim: int = 64, 
        latent_dim: int = 64,
        output_dim: int = 32, 
        device: Optional[str] = "cpu", 
        solver: Optional[str] = "dopri5", 
        sensitivity: Optional[str] = "adjoint",
        network_arch: Optional[nn.Sequential] = None,
    ):
        '''
        Params:
        - obs_space (gym.Space): observation space of the environment
        - features_dim (int): dimension of the features extracted
        - device (str, optional): device to run the model on, defaults to "cpu"
        - solver (str, optional): ODE solver to use, defaults to "dopri5"
        - sensitivity (str, optional): sensitivity method to use (i.e. how gradients for the ODE are backpropagated), defaults to "adjoint"
        - network_arch (nn.Sequential, optional): custom MLP architecture to use for the NODE, defaults to default network architecture in Policy class
        '''
        super(MlpLstmNodeExtractor, self).__init__(obs_space, output_dim)
        obs_dim = np.prod(obs_space.shape)

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.Tanh()
        )

        self.model = Policy(
            obs_space=obs_space, 
            device=device, 
            solver=solver, 
            sensitivity=sensitivity, 
            latent_dim=latent_dim,
            features_dim=features_dim,
            mlp_model=network_arch
        ).get_mlp_model()

        # Projection layer to get the desired feature dimension
        self.head = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            #nn.Linear(obs_dim, features_dim),
            nn.ReLU()
        )

        self.solver = solver
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        '''
        Params:
        - obs (torch.Tensor): input observation tensor

        Returns:
        - features (torch.Tensor): extracted feature tensor
        '''
        out = self.encoder(obs)
        # returns (ODE solution, t_eval integration interval) from the NODE output
        out = self.model(out)

        if isinstance(out, tuple):
            _, sol = out
        else:
            sol = out

        # returns 3D tensor (time, batch, features), take the last step for (batch, features)
        if sol.dim() == 3:
            sol = sol[-1]

        return self.head(sol.float())
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, output_dim: int = 32, latent_dim: int = 64) -> dict:
        '''
        Params:
        - features_dim (int): dimension of the features extracted

        Returns:
        - the policy kwargs (Dict) for Stable Baselines3 model initialization
        '''
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, output_dim=output_dim, latent_dim=latent_dim)
        )
    
class LSTMOutputExtractor(nn.Module):
    '''
    Class that extracts the output from an LSTM layer for Neural ODE inputs compatibility 
    '''
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, batch_first: bool = True):
        '''
        Params:
        - input_size (int): number of expected features in the input x
        - hidden_size (int): number of features in the hidden state h
        - num_layers (int): number of recurrent layers
        - batch_first (bool): if True, then the input and output tensors are provided as
        '''
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Params:
        - x (torch.Tensor): input tensor of shape (batch, seq_len, input_size

        Returns:
        - output (torch.Tensor): output tensor of shape (batch, seq_len, hidden_size
        '''
        # LSTM output => (output, (h_n, c_n))
        output, _ = self.lstm(x)
        return output