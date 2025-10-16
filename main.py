from model.model import train, SB3Agent, MlpNodeExtractor, CnnNodeExtractor, MlpLstmNodeExtractor
from stable_baselines3 import PPO
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="human")

    # NOTE: To change the model architecture, change the policy in `model/Policies.py`
    # Make sure to change the model_arch here to the corresponding feature extractor class.
    model_arch = MlpLstmNodeExtractor(
        obs_space=env.observation_space, 
        features_dim=64
    )

    agent = SB3Agent(sb3_class=PPO, model_path=None, model_arch=model_arch)
    train(agent, env)