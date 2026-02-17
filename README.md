# Continuum: A Dimension-Agnostic Neural ODE Deep Reinforcement Learning Framework for Physics-Based Environments
[[Project Page]](https://kseto06.github.io/Continuum/) | [[Paper]](https://kseto06.github.io/Continuum/assets/paper.pdf)

<p align="center">
    <img src="website/public/assets/architecture.png" alt="architecture">
</p>

**Kaden Seto**<sup>1</sup>, **Ryan Qian**<sup>1</sup>, **Kane Pan**<sup>1</sup>

<sup>1</sup>University of Toronto

In this work, we propose **Continuum**, a deep RL framework and neural network architecture for physics-informed reinforcement learning. The architecture combines NODEs, Autoencoders, and model-free RL algorithms, where the latent space of the Autoencoder is represented by a time-dependent NODE that learns the continuous-time dynamics of the environment. In this architecture, we aim to build a neural network that has stronger physics alignment and interpretability, thus encouraging policies to make predictions based on structured latent representations of the learned system dynamics that promote stability and performance.

---

## Installation, Setup and Usage

Follow the steps below to set up Continuum.

### Step 1: Clone the Repository
    
```sh
git clone https://github.com/kseto06/Continuum
```

### Step 2: Install Dependencies

1. Open a terminal and run the following set of commands to install necessary libraries and dependencies:

    Create a virtual environment:
    ```sh
    python -m venv venv
    ```

    If on macOS / Linux:
    ```sh
    source venv/bin/activate 
    ```

    If on Windows: 
    ```sh
    venv\Scripts\activate
    ```

    Install requirements:
    ```
    pip install -r requirements.txt
    ```

2. Once dependencies are installed, navigate to the correct directory:
    ```sh
    cd continuum
    ```

### Step 3: Run Training

Follow the instructions in `main.py` for setting up training:

1. Set the Gym/MuJoCo environment name out of the listed environments. Possible ClassicControl/Box2D/Mujoco envs to use that are physics-based are:
    - Classic Control: CartPole-v1, MountainCar-v0/MountainCarContinuous-v0, Acrobot-v1, Pendulum-v1, LunarLander-v2
    - Box2D: LunarLander-v3, BipedalWalker-v3
    - Mujoco: HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Humanoid-v5, HumanoidStandup-v5, Swimmer-v5, Reacher-v5, InvertedPendulum-v5, InvertedDoublePendulum-v5
    ```python
    env_name = "<env_name>"
    ```

2. If resuming training from a checkpoint, **set the path** of the `.pkl` file from the training checkpoint in `rl-model`:
    ```python
    env = VecNormalize.load("<path>", env)
    ```

   Otherwise, use the given default `VecNormalize` arguments provided, i.e:
    ```python
    env = make_vec_env(env_name, n_envs=16, vec_env_cls=SubprocVecEnv)
    ```

3. **Optionally**, set the model architecture using PyTorch's `nn.Sequential` class, otherwise the network architecture will be defaulted for you. An example of an architecture:
    ```python
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
    ```

4. Ensure that the corresponding architecture uses the correct `Extractor` class. If the network architecture is:
- Standard MLP: use `MlpNodeExtractor`
- CNN: use `CnnNodeExtractor`
- LSTM: use `MlpLstmNodeExtractor`
    If the network architecture in Step 3 was defaulted, then the `Extractor` class used will be automatically defaulted as well. 

5. For running training, run this command in terminal with the expected arguments:
    ```sh
    python main.py <solver_name (str)> <total_timesteps (int)> <checkpoint_interval (int)>
    ```
    
### Step 4: Run Inference

In `inference.py`:
1. Set the Gym/MuJoCo environment name to run inference on. We provide pretrained models on our NODE architecture and on standard PPO for `Humanoid-v5`, `HumanoidStandup-v5`, `Ant-v5`, and `HalfCheetah-v5`. 
    ```python
    env_name = "<env_name>"
    ```
2. Set the `.pkl` path and the model's path `.zip` to load the model. The file paths to our provided pretrained models are given below. For the `vec_path` and `model_path` variables, load ONLY either NODE files or PPO files:
- **Humanoid-v5**:
    - NODE `.pkl` Path: `model/rl-model/Humanoid-v5/Humanoid-v5_NODE_Pretrained.pkl`
    - NODE `.zip` Path: `model/rl-model/Humanoid-v5/Humanoid-v5_NODE_Pretrained.zip`
    - PPO `.pkl` Path: `model/rl-model/Humanoid-v5/Humanoid-v5_PPO_Pretrained.pkl`
    - PPO `.zip` Path: `model/rl-model/Humanoid-v5/Humanoid-v5_PPO_Pretrained.zip`
- **HumanoidStandup-v5**:
    - NODE `.pkl` Path: `model/rl-model/HumanoidStandup-v5/HumanoidStandup-v5_NODE_Pretrained.pkl`
    - NODE `.zip` Path: `model/rl-model/HumanoidStandup-v5/HumanoidStandup-v5_NODE_Pretrained.zip`
    - PPO `.pkl` Path: `model/rl-model/HumanoidStandup-v5/HumanoidStandup-v5_PPO_Pretrained.pkl`
    - PPO `.zip` Path: `model/rl-model/HumanoidStandup-v5/HumanoidStandup-v5_PPO_Pretrained.zip`
- **Ant-v5**: 
    - NODE `.pkl` Path: `model/rl-model/Ant-v5/Ant-v5_NODE_Pretrained.pkl`
    - NODE `.zip` Path: `model/rl-model/Ant-v5/Ant-v5_NODE_Pretrained.zip`
    - PPO `.pkl` Path: `model/rl-model/Ant-v5/Ant-v5_PPO_Pretrained.pkl`
    - PPO `.zip` Path: `model/rl-model/Ant-v5/Ant-v5_PPO_Pretrained.zip`
- **HalfCheetah-v5**:
    - NODE `.pkl` Path: `model/rl-model/HalfCheetah-v5/HalfCheetah-v5_NODE_Pretrained.pkl`
    - NODE `.zip` Path: `model/rl-model/HalfCheetah-v5/HalfCheetah-v5_NODE_Pretrained.zip`
    - PPO `.pkl` Path: `model/rl-model/HalfCheetah-v5/HalfCheetah-v5_PPO_Pretrained.pkl`
    - PPO `.zip` Path: `model/rl-model/HalfCheetah-v5/HalfCheetah-v5_PPO_Pretrained.zip`

    ```python
    vec_path = "<path_to_pkl>"
    model_path = "<path_to_model>"
    ```
3. Run the file in a terminal command:
    ```sh
    python inference.py
    ```
