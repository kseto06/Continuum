import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.model import train, SB3Agent
from stable_baselines3 import PPO
import gymnasium as gym

'''
Running this training code to train a regular SB3 agent for benchmarking requires only the file name for normal SB3 algoithm.

`python main.py <total_timesteps> <checkpoint_interval>`
'''

if __name__ == "__main__":
    # Possible classic control envs to use are:
    # CartPole-v1, MountainCar-v0/MountainCarContinuous-v0, Acrobot-v1, Pendulum-v1, LunarLander-v2
    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # env = gym.make("Acrobot-v1", render_mode="human")
    # env = gym.make("CartPole-v1", render_mode="human")
    # env = gym.make("Pendulum-v1", render_mode="human")
    env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")

    sb3_class = PPO
    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 128],
            vf=[128, 128],
        )
    )

    agent = SB3Agent(sb3_class=sb3_class, policy_kwargs=policy_kwargs, model_path=None, total_timesteps=int(sys.argv[1]), checkpoint_interval=int(sys.argv[2]))
    train(agent, env, run_name=f"{env.spec.id}-standard-{sb3_class.__name__}")