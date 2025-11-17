import gymnasium as gym
from stable_baselines3 import PPO, SAC

# env_name = "BipedalWalker-v3"
# model_path = "model/rl-model/NODE-PPO_rk4_BipedalWalker-v3_checkpoint_4834880_steps.zip"
env_name = "HumanoidStandup-v5"
model_path = "model/rl-model/NODE-PPO_rk4_HumanoidStandup-v5_checkpoint_1940936_steps.zip"
model = PPO.load(model_path)
env = gym.make(env_name, render_mode="human")
num_episodes = 50

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()

    print(f"Episode {episode + 1} reward: {total_reward}")

env.close()