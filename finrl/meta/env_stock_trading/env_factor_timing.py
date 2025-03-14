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

    def __init__(self, data, state_cols, factor_cols, initial_cash=1_000_000, render_mode='human'):
        super(FactorTimingEnv, self).__init__()
        self.data = data.sort_index()
        self.state_cols = state_cols
        self.factor_cols = factor_cols
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.current_step = 0
        self.num_factors = len(factor_cols)
        self.last_action = None  # Will store the most recent action
        self.render_mode = render_mode
        # Action space: allocations for each factor; allow long/short allocations
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_factors,), dtype=np.float32)
        # Observation space: state vector of economic indicators
        obs_dim = len(state_cols) + 1 + self.num_factors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, **kwargs):
        self.current_step = 0
        self.cash = self.initial_cash
        self.last_action = None
        return self._get_observation(), {}

    def _get_observation_old(self):
        row = self.data.iloc[self.current_step]
        return row[self.state_cols].values.astype(np.float32)

    def _get_observation(self):
        # Get market features from the current row
        row = self.data.iloc[self.current_step]
        market_features = row[self.state_cols].values.astype(np.float32)

        # Normalize cash by the initial cash to keep values in a manageable range
        cash_feature = np.array([self.cash / self.initial_cash], dtype=np.float32)

        # Include last action; if not available, use zeros
        if self.last_action is not None:
            last_action_feature = np.array(self.last_action, dtype=np.float32)
        else:
            last_action_feature = np.zeros(self.action_space.shape, dtype=np.float32)

        # Concatenate all features into one observation vector
        observation = np.concatenate([market_features, cash_feature, last_action_feature])

        # Optional: Ensure no NaNs are passed along
        observation = np.nan_to_num(observation, nan=0.0)
        return observation

    def step(self, action):
        # Ensure action is a numpy array and normalize it
        action = np.array(action)
        action = action / (np.sum(np.abs(action)) + 1e-8)

        # Store the previous action for trade cost calculation
        prev_action = self.last_action if self.last_action is not None else np.zeros_like(action)

        # Optional: set a transaction cost factor (adjust if needed)
        transaction_cost = 0.000  # For now, zero cost
        trade_cost = transaction_cost * np.sum(np.abs(action - prev_action))

        # Update the last action after computing trade cost
        self.last_action = action

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

        # Get factor returns for the current step (as decimals)
        factor_returns = self.data.iloc[self.current_step][self.factor_cols].values.astype(np.float32)

        # Calculate portfolio return as the dot product minus transaction costs
        portfolio_return = np.dot(action, factor_returns) - trade_cost

        # Update portfolio value (cash)
        self.cash *= (1 + portfolio_return)

        # Calculate reward as the log return, ensuring valid input with max()
        value = np.maximum(1 + portfolio_return, 1e-8)
        reward = np.log(value)

        self.current_step += 1
        terminated = self.current_step >= len(self.data)
        truncated = False  # Adjust if you use a time limit
        next_obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape,
                                                                           dtype=np.float32)

        info = {"cash": self.cash, "portfolio_return": portfolio_return, "action": action}
        return next_obs, reward, terminated, truncated, info

    def step_old(self, action):
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

    def step_for_predict(self, action):
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
        factor_returns = self.data.iloc[0][self.factor_cols].values.astype(np.float32)
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
        # Get the current date from the data index if available; otherwise, indicate the episode is over.
        if self.current_step < len(self.data):
            current_date = self.data.index[self.current_step]
        else:
            current_date = "End"
        print(
            f"Step: {self.current_step}, Date: {current_date}, Cash: {self.cash:.2f}, Last Action: {self.last_action}")
        return self.current_step, current_date, self.last_action




class FactorTimingEnvRandomStart(gym.Env):
    """
    A Gymnasium Environment for factor timing.
    - State: vector of economic indicators (state_cols)
    - Action: allocation vector for factors (continuous in [-1, 1])
    - Reward: portfolio return computed as the dot product of the allocation and the realized factor returns.

    This version supports:
      - A random starting position when the episode is reset.
      - A maximum episode length (max_episode_length) to truncate episodes.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, state_cols, factor_cols, initial_cash=1_000_000, max_episode_length=None,
                 render_mode=None):
        super(FactorTimingEnvRandomStart, self).__init__()
        self.data = data.sort_index()
        self.state_cols = state_cols
        self.factor_cols = factor_cols
        self.initial_cash = initial_cash
        self.max_episode_length = max_episode_length  # Maximum steps per episode
        self.cash = initial_cash
        self.num_factors = len(factor_cols)
        self.last_action = None  # Will store the most recent action
        self.random_start = 0  # Random starting index for the episode
        self.steps_taken = 0  # Counter for steps taken in the current episode
        self.render_mode = render_mode

        # Action space: allocations for each factor; allow long/short allocations
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_factors,), dtype=np.float32)
        # Observation space: state vector of economic indicators
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(state_cols),), dtype=np.float32)
        obs_dim = len(state_cols) + 1 + self.num_factors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    def reset(self, **kwargs):
        # Choose a random starting index if max_episode_length is defined and there is enough room.
        if self.max_episode_length is not None and len(self.data) > self.max_episode_length:
            self.random_start = np.random.randint(0, len(self.data) - self.max_episode_length)
        else:
            self.random_start = 0
        self.current_step = self.random_start
        self.cash = self.initial_cash
        self.steps_taken = 0
        self.last_action = None
        return self._get_observation(), {}

    def _get_observation_old(self):
        row = self.data.iloc[self.current_step]
        return row[self.state_cols].values.astype(np.float32)

    def _get_observation(self):
        # Get market features from the current row
        row = self.data.iloc[self.current_step]
        market_features = row[self.state_cols].values.astype(np.float32)
        normalized_cash = self.cash / self.initial_cash
        normalized_cash = np.clip(normalized_cash, 0, 1e6)
        cash_feature = np.array([normalized_cash], dtype=np.float32)
        # # Normalize cash by the initial cash to keep values in a manageable range
        # cash_feature = np.array([self.cash / self.initial_cash], dtype=np.float32)


        # Include last action; if not available, use zeros
        if self.last_action is not None:
            last_action_feature = np.array(self.last_action, dtype=np.float32)
        else:
            last_action_feature = np.zeros(self.action_space.shape, dtype=np.float32)

        # Concatenate all features into one observation vector
        observation = np.concatenate([market_features, cash_feature, last_action_feature])

        # Optional: Ensure no NaNs are passed along
        observation = np.nan_to_num(observation, nan=0.0)
        return observation

    def step(self, action):
        # Ensure action is a numpy array and normalize it
        action = np.array(action)
        action = action / (np.sum(np.abs(action)) + 1e-8)

        # Store the previous action for trade cost calculation
        prev_action = self.last_action if self.last_action is not None else np.zeros_like(action)

        # Optional: set a transaction cost factor (adjust if needed)
        transaction_cost = 0.000  # For now, zero cost
        trade_cost = transaction_cost * np.sum(np.abs(action - prev_action))

        # Update the last action after computing trade cost
        self.last_action = action

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

        # Get factor returns for the current step (as decimals)
        factor_returns = self.data.iloc[self.current_step][self.factor_cols].values.astype(np.float32)

        # Calculate portfolio return as the dot product minus transaction costs
        portfolio_return = np.dot(action, factor_returns) - trade_cost

        # Update portfolio value (cash)
        self.cash *= (1 + portfolio_return)

        # Calculate reward as the log return, ensuring valid input with max()
        value = np.maximum(1 + portfolio_return, 1e-8)
        reward = np.log(value)

        self.current_step += 1
        terminated = self.current_step >= len(self.data)
        truncated = False  # Adjust if you use a time limit
        next_obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape,
                                                                           dtype=np.float32)

        info = {"cash": self.cash, "portfolio_return": portfolio_return, "action": action}
        return next_obs, reward, terminated, truncated, info

    def step_old(self, action):
        # Check if we have reached the end of the data or the maximum episode length.
        if self.current_step >= len(self.data) or (
                self.max_episode_length is not None and self.steps_taken >= self.max_episode_length):
            terminated = self.current_step >= len(self.data)
            truncated = self.max_episode_length is not None and self.steps_taken >= self.max_episode_length
            return (np.zeros(self.observation_space.shape, dtype=np.float32),
                    0.0,
                    terminated,
                    truncated,
                    {"cash": self.cash, "portfolio_return": 0.0, "action": action})

        action = np.array(action)
        self.last_action = action  # Store last action

        # Get factor returns for the current step (as decimals)
        factor_returns = self.data.iloc[self.current_step][self.factor_cols].values.astype(np.float32)
        # Compute portfolio return as the dot product of action and factor returns.
        portfolio_return = np.dot(action, factor_returns)

        # Update cash value
        self.cash = self.cash * (1 + portfolio_return)

        # Compute reward as the log return (clipped to avoid log(0) issues)
        value = np.maximum(1 + portfolio_return, 1e-8)
        reward = np.log(value)

        self.current_step += 1
        self.steps_taken += 1

        terminated = self.current_step >= len(self.data)
        truncated = self.max_episode_length is not None and self.steps_taken >= self.max_episode_length
        next_obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape,
                                                                           dtype=np.float32)
        info = {"cash": self.cash, "portfolio_return": portfolio_return, "action": action}
        return next_obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.current_step < len(self.data):
            current_date = self.data.index[self.current_step]
        else:
            current_date = "End"
        print(
            f"Step: {self.current_step}, Date: {current_date}, Cash: {self.cash:.2f}, Last Action: {self.last_action}")
        return self.current_step, current_date, self.last_action
