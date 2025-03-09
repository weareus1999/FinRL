import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FactorTimingEnv(gym.Env):
    """
    A Gymnasium Environment for factor timing.
    - State: vector of economic indicators (state_cols)
    - Action: allocation vector for factors (continuous in [-1, 1])
    - Reward: portfolio return computed as the dot product of the allocation and the realized factor returns.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, state_cols, factor_cols, initial_cash=1_000_000):
        super(FactorTimingEnv, self).__init__()
        self.data = data.sort_index()
        self.state_cols = state_cols
        self.factor_cols = factor_cols
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.current_step = 0
        self.num_factors = len(factor_cols)
        self.last_action = None  # Will store the most recent action

        # Action space: allocations for each factor; allow long/short allocations
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_factors,), dtype=np.float32)
        # Observation space: state vector of economic indicators
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(state_cols),), dtype=np.float32)

    def reset(self, **kwargs):
        self.current_step = 0
        self.cash = self.initial_cash
        self.last_action = None
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        return row[self.state_cols].values.astype(np.float32)

    def step(self, action):
        # Check if current_step is out-of-bounds
        if self.current_step >= len(self.data):
            terminated = True
            truncated = False
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                terminated,
                truncated,
                {"cash": self.cash, "portfolio_return": 0.0, "action": action}
            )

        # Ensure action is a numpy array.
        action = np.array(action)
        self.last_action = action  # store last action

        # Get factor returns for current step (as decimals)
        factor_returns = self.data.iloc[self.current_step][self.factor_cols].values.astype(np.float32)
        # Calculate portfolio return as the dot product of the action and factor returns.
        portfolio_return = np.dot(action, factor_returns)

        # Update portfolio value (cash)
        self.cash = self.cash * (1 + portfolio_return)

        # Calculate reward as the log return (clipping to avoid log(0) or negative input)
        value = np.maximum(1 + portfolio_return, 1e-8)
        reward = np.log(value)

        self.current_step += 1
        terminated = self.current_step >= len(self.data)
        truncated = False  # You can modify this if you have a time limit
        next_obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {"cash": self.cash, "portfolio_return": portfolio_return, "action": action}
        return next_obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Cash: {self.cash:.2f}, Last Action: {self.last_action}")
