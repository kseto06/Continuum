import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

'''
Running the code requires only the file name, however ensure that the model paths and environments are set correctly in the code.

`python inference.py`
'''

'''
Set the Gym/MuJoCo environment name to run inference on.
We provide pretrained models on our NODE architecture and on standard PPO for:
- `Humanoid-v5`
- `HumanoidStandup-v5`
- `Ant-v5`
- `HalfCheetah-v5`

Currently, this file defaults to the `Humanoid-v5` environment.
'''
env_name = "Humanoid-v5"

def make_env():
    return gym.make(env_name, render_mode="human")

venv = DummyVecEnv([make_env])

'''
Set the `.pkl` path and the model's path `.zip` to load the model. 
The file paths to our provided pretrained models are given below.
For the `vec_path` and `model_path` variables, load ONLY either NODE files or PPO files:
- Humanoid-v5:
    - NODE `.pkl` Path: `model/rl-model/Humanoid-v5/Humanoid-v5_NODE_Pretrained.pkl`
    - NODE `.zip` Path: `model/rl-model/Humanoid-v5/Humanoid-v5_NODE_Pretrained.zip`
    - PPO `.pkl` Path: `model/rl-model/Humanoid-v5/Humanoid-v5_PPO_Pretrained.pkl`
    - PPO `.zip` Path: `model/rl-model/Humanoid-v5/Humanoid-v5_PPO_Pretrained.zip`
- HumanoidStandup-v5:
    - NODE `.pkl` Path: `model/rl-model/HumanoidStandup-v5/HumanoidStandup-v5_NODE_Pretrained.pkl`
    - NODE `.zip` Path: `model/rl-model/HumanoidStandup-v5/HumanoidStandup-v5_NODE_Pretrained.zip`
    - PPO `.pkl` Path: `model/rl-model/HumanoidStandup-v5/HumanoidStandup-v5_PPO_Pretrained.pkl`
    - PPO `.zip` Path: `model/rl-model/HumanoidStandup-v5/HumanoidStandup-v5_PPO_Pretrained.zip`
- Ant-v5: 
    - NODE `.pkl` Path: `model/rl-model/Ant-v5/Ant-v5_NODE_Pretrained.pkl`
    - NODE `.zip` Path: `model/rl-model/Ant-v5/Ant-v5_NODE_Pretrained.zip`
    - PPO `.pkl` Path: `model/rl-model/Ant-v5/Ant-v5_PPO_Pretrained.pkl`
    - PPO `.zip` Path: `model/rl-model/Ant-v5/Ant-v5_PPO_Pretrained.zip`
- HalfCheetah-v5:
    - NODE `.pkl` Path: `model/rl-model/HalfCheetah-v5/HalfCheetah-v5_NODE_Pretrained.pkl`
    - NODE `.zip` Path: `model/rl-model/HalfCheetah-v5/HalfCheetah-v5_NODE_Pretrained.zip`
    - PPO `.pkl` Path: `model/rl-model/HalfCheetah-v5/HalfCheetah-v5_PPO_Pretrained.pkl`
    - PPO `.zip` Path: `model/rl-model/HalfCheetah-v5/HalfCheetah-v5_PPO_Pretrained.zip`

Currently, this file defaults to the pretrained NODE model on the Humanoid-v5 environment.
'''
vec_path = "model/rl-model/Humanoid-v5/Humanoid-v5_NODE_Pretrained.pkl"
model_path = "model/rl-model/Humanoid-v5/Humanoid-v5_NODE_Pretrained.zip"

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