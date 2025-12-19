import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

'''
Running the code requires only the file name, however ensure that the model paths and environments are set correctly in the code.

`python inference.py`
'''

'''
Set the Gym/MuJoCo environment name to run inference on:
'''
env_name = "Ant-v5"

def make_env():
    return gym.make(env_name, render_mode="human")

venv = DummyVecEnv([make_env])

'''
Set the `.pkl` path and the model's path `.zip` to load the model:
'''
vec_path = "model/rl-model/ant/NODE-PPO_rk4_Ant-v5_checkpoint_30048256_steps_vecnormalize.pkl"
model_path = "model/rl-model/ant/NODE-PPO_rk4_Ant-v5_checkpoint_30048256_steps.zip"

vec_env = VecNormalize.load(vec_path, venv)
vec_env.training = False
vec_env.norm_reward = False
model = PPO.load(model_path, env=vec_env)

num_episodes = 50

try:
    for episode in range(num_episodes):
        
        obs = vec_env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward.item()
            vec_env.render()

        print(f"Episode {episode + 1}: reward = {total_reward}")
        
except KeyboardInterrupt:
    vec_env.close()

vec_env.close()