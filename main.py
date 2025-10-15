from model.model import train, SB3Agent, MlpNodeExtractor
from stable_baselines3 import PPO
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")

    model_arch = MlpNodeExtractor(
        obs_space=env.observation_space, 
        features_dim=64
    )

    agent = SB3Agent(sb3_class=PPO, model_path=None, model_arch=model_arch)
    train(agent, env)