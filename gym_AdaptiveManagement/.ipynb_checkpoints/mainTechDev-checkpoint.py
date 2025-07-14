import gymnasium as gym
import gymnasium as gym
from gymnasium.envs.registration import register
from techDevEnv import techDevEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Register the environment
register(
    id='Ecosystem-v0',  # Environment ID, used to make the environment
    entry_point='AdaptiveManagement:AdaptiveManagement',  # The entry point to your environment class
    max_episode_steps=100,  # Max number of steps per episode
)

# Create an instance of the environment
env = gym.make('Ecosystem-v0', Tmax=100, scenario='scenario_name', 
               K_min=0.2, K_max=1.0, delta_t_crit=2, sigmoid_bool=True)

model = PPO("MultiInputPolicy", env, verbose=0)
model.learn(total_timesteps=100)

# Use the trained model to choose actions
obs, _ = env.reset()  # Extract the observation from the reset tuple
done = False
truncated = False

while not done and not truncated:
    # Use the trained model to predict the action
    action, _states = model.predict(obs, deterministic=True)
    
    # Take the action in the environment
    obs, reward, done, truncated, info = env.step(action)
    
    # Render the environment
    env.render()