from model.model import train, SB3NodeAgent, MlpNodeExtractor, CnnNodeExtractor, MlpLstmNodeExtractor
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import sys

'''
Running the code requires the file name, solver parameter (String), total_timesteps parameter (int), checkpoint_interval (int) for the input

`python main.py <solver> <total_timesteps> <checkpoint_interval>`
'''

if __name__ == "__main__":
    '''
    This file trains in an arbituary Gym/Mujoco env. 
    - For training, use render_mode=None to save computational resources
    - For inference, use render_mode=human to visualize the env
    '''
    
    '''
    Set the environment name here.
    Possible ClassicControl/Box2D/Mujoco envs to use that are physics-based are:
    - Classic Control: CartPole-v1, MountainCar-v0/MountainCarContinuous-v0, Acrobot-v1, Pendulum-v1, LunarLander-v2
    - Box2D: LunarLander-v3, BipedalWalker-v3
    - Mujoco: HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Humanoid-v5, HumanoidStandup-v5, Swimmer-v5, Reacher-v5, InvertedPendulum-v5, InvertedDoublePendulum-v5
    '''
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    '''
    make_vec_env and VecNormalize are Gym functions to create vectorized environments and normalize observations/rewards. This allows for faster, efficient, stable training of agents.
    VecEnv parameters can be changed in the initialization of the environment.
    NOTE:
    - If training from scratch, initialize a VecNormalize wrapper. For example:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    - If resuming training from a checkpoint, load the VecNormalize wrapper from the saved file. For example:
        env = VecNormalize.load("<vec_normalize_pkl_path>", env)
    '''
    env = make_vec_env(env_name, n_envs=16, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    # env = VecNormalize.load("<path>", env)

    network_arch = None
    solver = sys.argv[1]

    '''
    # NOTE: 
    # Optionally change the model architecture using nn.Sequential, otherwise it is defaulted in Policies.py. 
    # Example of defining a MLP-LSTM-NODE architecture below:

    latent_dim = 64
    features_dim = 128
    network_arch = nn.Sequential(
        DepthCat(1),
        nn.Linear(latent_dim + 1, features_dim),
        nn.Tanh(),
        LSTMOutputExtractor(input_size=features_dim, hidden_size=features_dim, num_layers=1, batch_first=True),
        nn.Tanh(),
        nn.Linear(features_dim, latent_dim)
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
        features_dim=128,
        latent_dim=64,
        output_dim=32,
        device="cpu",
        solver=solver, # change to euler, rk4, dormand prince, etc. specify in command line argument
        sensitivity="adjoint",
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
    model_path = None
    agent = SB3NodeAgent(env_name=env_name, sb3_class=sb3_class, model_path=model_path, model_arch=model_arch, total_timesteps=int(sys.argv[2]), checkpoint_interval=int(sys.argv[3]))

    train(agent, env, run_name=f"{env_name}-{sb3_class.__name__}-node-{solver}")