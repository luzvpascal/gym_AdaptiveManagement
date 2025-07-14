import gymnasium as gym
from gymnasium import spaces
import numpy as np

class techDevEnv(gym.Env):
    def __init__(self, Tmax, scenario, K_min, K_max, delta_t_crit, sigmoid_bool):
        super(techDevEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: BAU, 1: R&D, 2: Deploy

        self.observation_space = spaces.Dict({
            'ecosystem': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'temperature': spaces.Box(low=1, high=6, shape=(1,), dtype=np.float32),
            'time': spaces.Discrete(Tmax + 1),
            'technology': spaces.Discrete(2)  # 0: idle, 1: ready
        })

        # Set the environment parameters
        self.Tmax = Tmax #max number of time steps
        self.scenario = scenario #climate change scenario
        self.K_min = K_min #minimum capacity
        self.K_max = K_max #maximum capacity
        self.delta_t_crit = delta_t_crit #critical temperature (for K function)
        self.sigmoid_bool = sigmoid_bool #boolean indicating if sigmoid K or not

        # Initialize state
        self.state = {
            'ecosystem': 1.0,
            'temperature': 1.0,
            'time': 0,
            'technology': 0
        }

        # Load temperature data from the dataset
        self.temperature_series = self.load_temperature_series()

    def load_temperature_series(self):
        # Load or generate the temperature time series based on the scenario
        # For simplicity, we'll use a placeholder
        return np.linspace(1, 6, self.Tmax + 1)

    def ecosystem_dynamics(self, x_t, r, K, time_step):
        return x_t + time_step * (x_t * r * (1 - x_t / K))

    def K_function(self, K_min, K_max, delta_t, delta_t_crit, sigmoid_bool):
        if sigmoid_bool:
            return K_min + (K_max - K_min) * (1 - 1 / (1 + np.exp(-5 * (delta_t - delta_t_crit))))
        else:
            return K_max

    def step(self, action):
        x_t = self.state['ecosystem']
        r = 0.05  # Example intrinsic growth rate
        time_step = 1
        self.state['temperature'] = self.temperature_series[self.state['time']]
        delta_t = self.state['temperature'] - self.delta_t_crit
        K = self.K_function(self.K_min, self.K_max, delta_t, self.delta_t_crit, self.sigmoid_bool)

        if action == 0:  # BAU
            pass
        elif action == 1:  # R&D
            if np.random.rand() < 0.1:
                self.state['technology'] = 1  # Transition to ready
        elif action == 2 and self.state['technology'] == 1:  # Deploy
            K = self.K_function(self.K_min, self.K_max, delta_t-1, self.delta_t_crit, self.sigmoid_bool)

        # Update ecosystem state
        self.state['ecosystem'] = self.ecosystem_dynamics(x_t, r, K, time_step)

        # Increment time
        self.state['time'] += 1

        # Reward calculation
        reward = self.state['ecosystem']
        if action == 1:
            reward -= 0.05  # Cost of R&D
        elif action == 2:
            reward -= 0.05  # Cost of deployment

        # Check if the episode is done
        done = self.state['time'] >= self.Tmax
        truncated = done
        obs = {
        'ecosystem': np.array([self.state['ecosystem']], dtype=np.float32),
        'temperature': np.array([self.state['temperature']], dtype=np.float32),
        'time': np.array([self.state['time']], dtype=int),
        'technology': np.array([self.state['technology']],dtype=int)
        }
        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        # Handle the seed argument to allow reproducibility
        super().reset(seed=seed)
        np.random.seed(seed)

        self.state = {
            'ecosystem': 1.0,
            'temperature': 1.0,
            'time': 0,
            'technology': 0
        }
        
        obs = {
        'ecosystem': np.array([self.state['ecosystem']], dtype=np.float32),
        'temperature': np.array([self.state['temperature']], dtype=np.float32),
        'time': np.array([self.state['time']], dtype=int),
        'technology': np.array([self.state['technology']],dtype=int)
        }
        return obs, {}


    def render(self, mode='human'):
        print(f"Time: {self.state['time']}, Ecosystem: {self.state['ecosystem']:.2f}, "
              f"Temperature: {self.state['temperature']:.2f}, Technology: {'Ready' if self.state['technology'] == 1 else 'Idle'}")

