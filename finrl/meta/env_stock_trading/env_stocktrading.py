from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.signal import argrelextrema
import numpy as np
from ta.volatility import BollingerBands

import random


matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        sharpe_factor=0.5,
        returns_factor=0.5,
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()
        self.sharpe_factor = sharpe_factor
        self.returns_factor = returns_factor
        self.short_term_factor = 0.0  # Bonus factor for immediate profit
        self.activity_factor = 0.001  # Bonus factor for trading activity (sum of absolute trades)
        self.inactivity_penalty = 0.1



    def detect_peaks(df, column="close", order=5):
        df["local_max"] = df[column][argrelextrema(df[column].values, np.greater_equal, order=order)[0]]
        df["local_min"] = df[column][argrelextrema(df[column].values, np.less_equal, order=order)[0]]
        df["is_peak"] = df["local_max"].notnull().astype(int)  # Convert peaks to binary indicator
        df["is_valley"] = df["local_min"].notnull().astype(int)
        return df

    def fractal_indicator(df):
        df["fractal_up"] = ((df["high"].shift(2) < df["high"].shift(1)) &
                            (df["high"].shift(1) < df["high"]) &
                            (df["high"] > df["high"].shift(-1)) &
                            (df["high"].shift(-1) > df["high"].shift(-2))).astype(int)

        df["fractal_down"] = ((df["low"].shift(2) > df["low"].shift(1)) &
                              (df["low"].shift(1) > df["low"]) &
                              (df["low"] < df["low"].shift(-1)) &
                              (df["low"].shift(-1) < df["low"].shift(-2))).astype(int)
        return df

    def add_bollinger_band_width(df):
        indicator_bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_width"] = indicator_bb.bollinger_hband() - indicator_bb.bollinger_lband()
        return df

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()


    def steporig(self, actions):
        #print(f"Day {self.day}: Actions Taken: {actions}")
        def compute_sharpe_ratio(returns):
            return np.mean(returns) / (np.std(returns) + 1e-9)
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            return self.state, self.reward, self.terminal, False, {}

        actions = actions * self.hmax  # Scale actions
        actions = actions.astype(int)  # Convert to integers (can't buy fractional shares)

        # Calculate initial portfolio value
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )

        # Execute buy/sell actions
        for index in np.argsort(actions)[:np.where(actions < 0)[0].shape[0]]:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)
        for index in np.argsort(actions)[::-1][:np.where(actions > 0)[0].shape[0]]:
            actions[index] = self._buy_stock(index, actions[index])

        # Move to next time step
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        # Calculate final portfolio value
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )

        #Get peak signals
        is_peak = self.data["is_peak"]
        is_valley = self.data["is_valley"]
        # Reward for selling at peaks
        sell_reward = sum(actions * is_peak * self.state[1:self.stock_dim + 1])

        # Reward for buying at valleys
        buy_reward = sum(-actions * is_valley * self.state[1:self.stock_dim + 1])

        # Calculate profit/loss per trade (instead of just total asset change)
        trade_profit = end_total_asset - begin_total_asset
        trading_cost_penalty = self.cost * 0.01  # Penalize transaction costs
        reward = trade_profit - trading_cost_penalty  # Adjusted reward

        # Normalize reward using Sharpe Ratio
        if len(self.rewards_memory) > 1:
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            sharpe_reward = compute_sharpe_ratio(returns)
        else:
            sharpe_reward = 0

        # Final reward calculation (mix of profit and risk-adjusted return)
        portfolio_reward = self.returns_factor * reward + self.sharpe_factor * sharpe_reward
        self.reward = portfolio_reward + 0.1 * sell_reward + 0.1 * buy_reward
        self.reward = self.reward * self.reward_scaling  # Apply scaling factor

        # Store memory for future use
        self.asset_memory.append(end_total_asset)
        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)
        self.date_memory.append(self._get_date())
        self.actions_memory.append(actions.tolist())

        return self.state, self.reward, self.terminal, False, {}

    def step(self, actions):
        def compute_sharpe_ratio(returns):
            return np.mean(returns) / (np.std(returns) + 1e-9)

        # Check for terminal condition
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            return self.state, self.reward, self.terminal, False, {}

        # Scale and convert actions
        actions = actions * self.hmax
        actions = actions.astype(int)
        actions = np.atleast_1d(actions)

        # Calculate initial portfolio value (cash + stock value)
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1:(self.stock_dim + 1)]) *
            np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
        )

        # Execute sell actions (for negative actions)
        for index in np.argsort(actions)[:np.where(actions < 0)[0].shape[0]]:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)
        # Execute buy actions (for positive actions)
        for index in np.argsort(actions)[::-1][:np.where(actions > 0)[0].shape[0]]:
            actions[index] = self._buy_stock(index, actions[index])

        # Move to next day
        self.day += 1

        # Check if we've exceeded the data length
        if self.day >= len(self.df):
            self.terminal = True
            self.data = self.df.iloc[-1]
            self.state = self._update_state()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)]) *
                np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )
            final_reward = 0  # Final reward; can adjust if needed
            return self.state, final_reward, self.terminal, False, {}

        # Update data and state for the new day
        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        # Calculate final portfolio value for the current day
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1:(self.stock_dim + 1)]) *
            np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
        )

        # Get signals from technical indicators
        is_peak = self.data["is_peak"]
        is_valley = self.data["is_valley"]
        sell_reward = sum(actions * is_peak * self.state[1:(self.stock_dim + 1)])
        buy_reward = sum(-actions * is_valley * self.state[1:(self.stock_dim + 1)])

        # Compute immediate trade profit (short-term profit)
        trade_profit = end_total_asset - begin_total_asset
        trading_cost_penalty = self.cost * 0.01  # You might already have cost set to zero
        immediate_reward = trade_profit - trading_cost_penalty

        # Compute risk-adjusted reward using Sharpe ratio (over previous days)
        if len(self.rewards_memory) > 1:
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            sharpe_reward = compute_sharpe_ratio(returns)
        else:
            sharpe_reward = 0

        # Combine rewards: original portfolio reward, plus short-term bonus and activity bonus
        portfolio_reward = self.returns_factor * immediate_reward + self.sharpe_factor * sharpe_reward

        # Short-term reward bonus (emphasize immediate profit)
        short_term_reward = self.short_term_factor * trade_profit
        # Activity bonus: reward proportional to the total absolute trade amount
        activity_bonus = self.activity_factor * np.sum(np.abs(actions))
        # Inactivity penalty: if no trades occur, apply a small penalty
        inactivity_penalty = 0 if np.sum(np.abs(actions)) > 0 else self.inactivity_penalty

        # Final reward: sum all components plus a small bonus for peak/valley actions
        self.reward = (portfolio_reward + 0.1 * sell_reward + 0.1 * buy_reward +
                       short_term_reward + activity_bonus - inactivity_penalty)
        self.reward = self.reward * self.reward_scaling
        self.reward = np.clip(self.reward, -10, 10)

        # Store memory for later analysis
        self.asset_memory.append(end_total_asset)
        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)
        self.date_memory.append(self._get_date())
        self.actions_memory.append(actions.tolist())

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        """
        Saves the action memory as a Pandas DataFrame while ensuring
        `date_memory` and `actions_memory` have matching lengths.
        """

        # Ensure date_list and actions_memory have the same length
        min_length = min(len(self.date_memory), len(self.actions_memory))

        # If no actions were taken, return an empty DataFrame
        if min_length == 0:
            return pd.DataFrame(columns=["date", "actions"])

        # Trim lists to match lengths
        date_list = self.date_memory[:min_length]
        action_list = self.actions_memory[:min_length]

        if len(self.df.tic.unique()) > 1:  # Multiple stocks case
            df_date = pd.DataFrame({"date": date_list})
            df_actions = pd.DataFrame(action_list)

            # Ensure correct column assignment
            if df_actions.shape[1] == len(self.data.tic.values):
                df_actions.columns = self.data.tic.values
            else:
                df_actions.columns = [f"Stock_{i}" for i in range(df_actions.shape[1])]

            df_actions.index = df_date["date"]
        else:  # Single stock case
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})

        print(f"🔹 Saved {len(df_actions)} action records.")  # Debugging output
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

class StockTradingEnvRandomStarts(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            df: pd.DataFrame,
            stock_dim: int,
            hmax: int,
            initial_amount: int,
            num_stock_shares: list,
            buy_cost_pct: list,
            sell_cost_pct: list,
            reward_scaling: float,
            state_space: int,
            action_space: int,
            tech_indicator_list: list,
            turbulence_threshold=None,
            risk_indicator_col="turbulence",
            make_plots: bool = False,
            print_verbosity: int = 10,
            initial: bool = True,
            previous_state: list = [],
            model_name: str = "",
            mode: str = "",
            iteration: str = "",
            sharpe_factor: float = 0.5,
            returns_factor: float = 0.5,
            max_episode_length: int = 90,  # number of days per episode
    ):
        super().__init__()
        self.df = df.copy()
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space_dim = action_space
        self.tech_indicator_list = tech_indicator_list

        # Define action and observation spaces.
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32)

        # Store sorted unique dates (as strings or numbers, as in your df)
        self.unique_days = np.array(sorted(self.df["date"].unique()))
        print("UNQEDAYS", len(self.unique_days))
        self.max_episode_length = max_episode_length

        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.sharpe_factor = sharpe_factor
        self.returns_factor = returns_factor
        self.short_term_factor = 0.0
        self.activity_factor = 0.001
        self.inactivity_penalty = 0.1

        self._seed()

        # These variables will be set/reset in reset()
        self.day = 0
        self.start_day = 0  # index in unique_days for the episode start
        self.end_day = 0  # start_day + max_episode_length
        self.data = None
        self.terminal = False
        self.episode = 0

        # Initialize state variables
        self.state = None
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0

        # Memory arrays
        self.asset_memory = []
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = []

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()


    def steporig(self, actions):
        #print(f"Day {self.day}: Actions Taken: {actions}")
        def compute_sharpe_ratio(returns):
            return np.mean(returns) / (np.std(returns) + 1e-9)
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            return self.state, self.reward, self.terminal, False, {}

        actions = actions * self.hmax  # Scale actions
        actions = actions.astype(int)  # Convert to integers (can't buy fractional shares)

        # Calculate initial portfolio value
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )

        # Execute buy/sell actions
        for index in np.argsort(actions)[:np.where(actions < 0)[0].shape[0]]:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)
        for index in np.argsort(actions)[::-1][:np.where(actions > 0)[0].shape[0]]:
            actions[index] = self._buy_stock(index, actions[index])

        # Move to next time step
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        # Calculate final portfolio value
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )

        #Get peak signals
        is_peak = self.data["is_peak"]
        is_valley = self.data["is_valley"]
        # Reward for selling at peaks
        sell_reward = sum(actions * is_peak * self.state[1:self.stock_dim + 1])

        # Reward for buying at valleys
        buy_reward = sum(-actions * is_valley * self.state[1:self.stock_dim + 1])

        # Calculate profit/loss per trade (instead of just total asset change)
        trade_profit = end_total_asset - begin_total_asset
        trading_cost_penalty = self.cost * 0.01  # Penalize transaction costs
        reward = trade_profit - trading_cost_penalty  # Adjusted reward

        # Normalize reward using Sharpe Ratio
        if len(self.rewards_memory) > 1:
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            sharpe_reward = compute_sharpe_ratio(returns)
        else:
            sharpe_reward = 0

        # Final reward calculation (mix of profit and risk-adjusted return)
        portfolio_reward = self.returns_factor * reward + self.sharpe_factor * sharpe_reward
        self.reward = portfolio_reward + 0.1 * sell_reward + 0.1 * buy_reward
        self.reward = self.reward * self.reward_scaling  # Apply scaling factor

        # Store memory for future use
        self.asset_memory.append(end_total_asset)
        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)
        self.date_memory.append(self._get_date())
        self.actions_memory.append(actions.tolist())

        return self.state, self.reward, self.terminal, False, {}

    def step(self, actions):
        """Take an action for one day, update state and reward, and move forward one day."""
        def compute_sharpe_ratio(returns):
            return np.mean(returns) / (np.std(returns) + 1e-9)

        # Determine the day offset (how many days have passed in this episode)
        current_date = str(self._get_date())
        unique_days_str = np.array([str(d) for d in self.unique_days])
        day_index_in_unique = np.where(unique_days_str == current_date)[0][0]
        day_offset = day_index_in_unique - self.start_day

        if day_offset >= self.max_episode_length - 1 or self.day >= self.df.index[-1]:
            return self.state, 0.0, True, False, {}

        # Scale and convert actions
        actions = actions * self.hmax
        actions = actions.astype(int)
        actions = np.atleast_1d(actions)

        # Compute current portfolio value
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1:1+self.stock_dim]) *
            np.array(self.state[self.stock_dim+1:self.stock_dim*2+1])
        )

        # Execute sell actions for negative actions
        for index in np.argsort(actions)[: np.where(actions < 0)[0].shape[0]]:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)
        # Execute buy actions for positive actions
        for index in np.argsort(actions)[::-1][: np.where(actions > 0)[0].shape[0]]:
            actions[index] = self._buy_stock(index, actions[index])

        # Move to next day
        self.day += 1
        if self.day >= len(self.df):
            return self.state, 0.0, True, False, {}

        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        # Compute new portfolio value
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1:1+self.stock_dim]) *
            np.array(self.state[self.stock_dim+1:self.stock_dim*2+1])
        )
        trade_profit = end_total_asset - begin_total_asset
        trading_cost_penalty = self.cost * 0.01
        immediate_reward = trade_profit - trading_cost_penalty

        if len(self.rewards_memory) > 1:
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            sharpe_reward = compute_sharpe_ratio(returns)
        else:
            sharpe_reward = 0

        portfolio_reward = self.returns_factor * immediate_reward + self.sharpe_factor * sharpe_reward
        # (You can add additional bonus terms here if desired)
        reward = portfolio_reward
        reward = reward * self.reward_scaling
        reward = np.clip(reward, -10, 10)
        self.reward = reward

        # Store memory
        self.asset_memory.append(end_total_asset)
        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)
        self.date_memory.append(self._get_date())
        self.actions_memory.append(actions.tolist())

        # Check if we have reached max_episode_length
        current_date_new = str(self._get_date())

        day_index_new = np.where(unique_days_str == current_date_new)[0][0]
        done = (day_index_new - self.start_day) >= (self.max_episode_length - 1)

        return self.state, self.reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        """Randomize the start day and reset the environment for a new episode."""
        # Randomly choose a starting day index such that there are enough days left for an episode.
        max_start = len(self.unique_days) - self.max_episode_length
        self.start_day = random.randint(0, max_start)
        self.end_day = self.start_day + self.max_episode_length

        # Get the start date
        start_date = self.unique_days[self.start_day]
        # Find the first row in df with this start date
        start_rows = self.df[self.df["date"] == start_date]
        if len(start_rows) == 0:
            raise ValueError("No rows found for start_date. Check your data.")
        self.day = start_rows.index[0]

        # Reinitialize episode variables
        self.terminal = False
        self.episode += 1
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0

        if self.initial:
            initial_asset = self.initial_amount + np.sum(
                np.array(self.num_stock_shares) * np.array(self.state[1:1+self.stock_dim])
            )
            self.asset_memory = [initial_asset]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1:1+self.stock_dim]) *
                np.array(self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)])
            )
            self.asset_memory = [previous_total_asset]

        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        """Build the initial state vector from the current row of data."""
        if self.initial:
            if len(self.df.tic.unique()) > 1:
                state = (
                        [self.initial_amount]
                        + self.data.close.values.tolist()
                        + self.num_stock_shares
                        + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])
                )
            else:
                state = (
                        [self.initial_amount]
                        + [self.data.close]
                        + [0] * self.stock_dim
                        + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            if len(self.df.tic.unique()) > 1:
                state = (
                        [self.previous_state[0]]
                        + self.data.close.values.tolist()
                        + self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]
                        + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])
                )
            else:
                state = (
                        [self.previous_state[0]]
                        + [self.data.close]
                        + self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]
                        + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        """Update the state vector for the new day."""
        if len(self.df.tic.unique()) > 1:
            state = (
                    [self.state[0]] +
                    self.data.close.values.tolist() +
                    list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) +
                    sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), [])
            )
        else:
            state = (
                    [self.state[0]] +
                    [self.data.close] +
                    list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) +
                    sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )
        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        """
        Saves the action memory as a Pandas DataFrame while ensuring
        `date_memory` and `actions_memory` have matching lengths.
        """

        # Ensure date_list and actions_memory have the same length
        min_length = min(len(self.date_memory), len(self.actions_memory))

        # If no actions were taken, return an empty DataFrame
        if min_length == 0:
            return pd.DataFrame(columns=["date", "actions"])

        # Trim lists to match lengths
        date_list = self.date_memory[:min_length]
        action_list = self.actions_memory[:min_length]

        if len(self.df.tic.unique()) > 1:  # Multiple stocks case
            df_date = pd.DataFrame({"date": date_list})
            df_actions = pd.DataFrame(action_list)

            # Ensure correct column assignment
            if df_actions.shape[1] == len(self.data.tic.values):
                df_actions.columns = self.data.tic.values
            else:
                df_actions.columns = [f"Stock_{i}" for i in range(df_actions.shape[1])]

            df_actions.index = df_date["date"]
        else:  # Single stock case
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})

        print(f"🔹 Saved {len(df_actions)} action records.")  # Debugging output
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


class StockTradingEnvWithGNN(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        sharpe_factor=0.5,
        returns_factor=0.5,
        embeddings=None,
        embedding_dim=16,
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        if embeddings is None:
            print("⚠️ Warning: Embeddings not provided! Defaulting to zero vectors.")
            self.embeddings = {tic: [0] * 16 for tic in self.df.tic.unique()}  # Assume embedding size = 16
        else:
            self.embeddings = embeddings  # Store stock embeddings
        self.embedding_dim = embedding_dim
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()
        self.sharpe_factor = sharpe_factor
        self.returns_factor = returns_factor
        self.short_term_factor = 0.0  # Bonus factor for immediate profit
        self.activity_factor = 0.001  # Bonus factor for trading activity (sum of absolute trades)
        self.inactivity_penalty = 0.1




    def detect_peaks(df, column="close", order=5):
        df["local_max"] = df[column][argrelextrema(df[column].values, np.greater_equal, order=order)[0]]
        df["local_min"] = df[column][argrelextrema(df[column].values, np.less_equal, order=order)[0]]
        df["is_peak"] = df["local_max"].notnull().astype(int)  # Convert peaks to binary indicator
        df["is_valley"] = df["local_min"].notnull().astype(int)
        return df

    def fractal_indicator(df):
        df["fractal_up"] = ((df["high"].shift(2) < df["high"].shift(1)) &
                            (df["high"].shift(1) < df["high"]) &
                            (df["high"] > df["high"].shift(-1)) &
                            (df["high"].shift(-1) > df["high"].shift(-2))).astype(int)

        df["fractal_down"] = ((df["low"].shift(2) > df["low"].shift(1)) &
                              (df["low"].shift(1) > df["low"]) &
                              (df["low"] < df["low"].shift(-1)) &
                              (df["low"].shift(-1) < df["low"].shift(-2))).astype(int)
        return df

    def add_bollinger_band_width(df):
        indicator_bb = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_width"] = indicator_bb.bollinger_hband() - indicator_bb.bollinger_lband()
        return df

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()


    def step(self, actions):
        #print(f"Day {self.day}: Actions Taken: {actions}")
        def compute_sharpe_ratio(returns):
            return np.mean(returns) / (np.std(returns) + 1e-9)
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            return self.state, self.reward, self.terminal, False, {}

        actions = actions * self.hmax  # Scale actions
        actions = actions.astype(int)  # Convert to integers (can't buy fractional shares)

        # Calculate initial portfolio value
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )

        # Execute buy/sell actions
        for index in np.argsort(actions)[:np.where(actions < 0)[0].shape[0]]:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)
        for index in np.argsort(actions)[::-1][:np.where(actions > 0)[0].shape[0]]:
            actions[index] = self._buy_stock(index, actions[index])

        # Move to next time step
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        # Calculate final portfolio value
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )

        #Get peak signals
        is_peak = self.data["is_peak"]
        is_valley = self.data["is_valley"]
        # Reward for selling at peaks
        sell_reward = sum(actions * is_peak * self.state[1:self.stock_dim + 1])

        # Reward for buying at valleys
        buy_reward = sum(-actions * is_valley * self.state[1:self.stock_dim + 1])

        # Calculate profit/loss per trade (instead of just total asset change)
        trade_profit = end_total_asset - begin_total_asset
        trading_cost_penalty = self.cost * 0.01  # Penalize transaction costs
        reward = trade_profit - trading_cost_penalty  # Adjusted reward

        # Normalize reward using Sharpe Ratio
        if len(self.rewards_memory) > 1:
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            sharpe_reward = compute_sharpe_ratio(returns)
        else:
            sharpe_reward = 0

        # Final reward calculation (mix of profit and risk-adjusted return)
        portfolio_reward = self.returns_factor * reward + self.sharpe_factor * sharpe_reward
        self.reward = portfolio_reward + 0.1 * sell_reward + 0.1 * buy_reward
        self.reward = self.reward * self.reward_scaling  # Apply scaling factor

        # Store memory for future use
        self.asset_memory.append(end_total_asset)
        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)
        self.date_memory.append(self._get_date())
        self.actions_memory.append(actions.tolist())

        return self.state, self.reward, self.terminal, False, {}

    def stepencourageactivity(self, actions):
        def compute_sharpe_ratio(returns):
            return np.mean(returns) / (np.std(returns) + 1e-9)

        # Check for terminal condition
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            return self.state, self.reward, self.terminal, False, {}

        # Scale and convert actions
        actions = actions * self.hmax
        actions = actions.astype(int)
        actions = np.atleast_1d(actions)

        # Calculate initial portfolio value (cash + stock value)
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1:(self.stock_dim + 1)]) *
            np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
        )

        # Execute sell actions (for negative actions)
        for index in np.argsort(actions)[:np.where(actions < 0)[0].shape[0]]:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)
        # Execute buy actions (for positive actions)
        for index in np.argsort(actions)[::-1][:np.where(actions > 0)[0].shape[0]]:
            actions[index] = self._buy_stock(index, actions[index])

        # Move to next day
        self.day += 1

        # Check if we've exceeded the data length
        if self.day >= len(self.df):
            self.terminal = True
            self.data = self.df.iloc[-1]
            self.state = self._update_state()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)]) *
                np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )
            final_reward = 0  # Final reward; can adjust if needed
            return self.state, final_reward, self.terminal, False, {}

        # Update data and state for the new day
        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        # Calculate final portfolio value for the current day
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1:(self.stock_dim + 1)]) *
            np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
        )

        # Get signals from technical indicators
        is_peak = self.data["is_peak"]
        is_valley = self.data["is_valley"]
        sell_reward = sum(actions * is_peak * self.state[1:(self.stock_dim + 1)])
        buy_reward = sum(-actions * is_valley * self.state[1:(self.stock_dim + 1)])

        # Compute immediate trade profit (short-term profit)
        trade_profit = end_total_asset - begin_total_asset
        trading_cost_penalty = self.cost * 0.01  # You might already have cost set to zero
        immediate_reward = trade_profit - trading_cost_penalty

        # Compute risk-adjusted reward using Sharpe ratio (over previous days)
        if len(self.rewards_memory) > 1:
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            sharpe_reward = compute_sharpe_ratio(returns)
        else:
            sharpe_reward = 0

        # Combine rewards: original portfolio reward, plus short-term bonus and activity bonus
        portfolio_reward = self.returns_factor * immediate_reward + self.sharpe_factor * sharpe_reward

        # Short-term reward bonus (emphasize immediate profit)
        short_term_reward = self.short_term_factor * trade_profit
        # Activity bonus: reward proportional to the total absolute trade amount
        activity_bonus = self.activity_factor * np.sum(np.abs(actions))
        # Inactivity penalty: if no trades occur, apply a small penalty
        inactivity_penalty = 0 if np.sum(np.abs(actions)) > 0 else self.inactivity_penalty

        # Final reward: sum all components plus a small bonus for peak/valley actions
        self.reward = (portfolio_reward + 0.1 * sell_reward + 0.1 * buy_reward +
                       short_term_reward + activity_bonus - inactivity_penalty)
        self.reward = self.reward * self.reward_scaling

        # Store memory for later analysis
        self.asset_memory.append(end_total_asset)
        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)
        self.date_memory.append(self._get_date())
        self.actions_memory.append(actions.tolist())

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        """Initialize state with stock embeddings."""

        # Get stock embeddings (assumed to be a dictionary: {tic: embedding_vector})
        stock_embeddings = self.embeddings

        if self.initial:
            # Initial State (first day)
            if len(self.df.tic.unique()) > 1:
                # Multiple stocks
                state = (
                        [self.initial_amount]  # Cash balance
                        + self.data.close.values.tolist()  # Closing prices
                        + self.num_stock_shares  # Initial shares owned
                        + sum(
                    (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                    [],
                )  # Technical indicators
                )

                # Append stock embeddings
                for tic in self.data.tic.unique():
                    state.extend(stock_embeddings.get(tic, [0] * self.embedding_dim))  # Default to zeros

            else:
                # Single stock case
                tic = self.data.tic.iloc[0]
                state = (
                        [self.initial_amount]
                        + [self.data.close]
                        + [0] * self.stock_dim  # Initial shares are zero
                        + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                        + stock_embeddings.get(tic, [0] * self.embedding_dim)  # Add embedding
                )

        else:
            # Use Previous State
            if len(self.df.tic.unique()) > 1:
                # Multiple stocks
                state = (
                        [self.previous_state[0]]  # Cash balance
                        + self.data.close.values.tolist()
                        + self.previous_state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]  # Previous stock holdings
                        + sum(
                    (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                    [],
                )
                )

                # Append stock embeddings
                for tic in self.data.tic.unique():
                    state.extend(stock_embeddings.get(tic, [0] * self.embedding_dim))

            else:
                # Single stock case
                tic = self.data.tic.iloc[0]
                state = (
                        [self.previous_state[0]]
                        + [self.data.close]
                        + self.previous_state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
                        + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                        + stock_embeddings.get(tic, [0] * self.embedding_dim)  # Add embedding
                )

        return state

    def _update_state(self):
        """Update the state representation with stock embeddings."""

        stock_embeddings = self.embeddings

        if len(self.df.tic.unique()) > 1:
            # Multiple stocks
            state = (
                    [self.state[0]]  # Cash balance
                    + self.data.close.values.tolist()  # Closing prices
                    + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])  # Stock holdings
                    + sum(
                (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                [],
            )  # Technical indicators
            )

            # Append stock embeddings
            for tic in self.data.tic.unique():
                state.extend(stock_embeddings.get(tic, [0] * self.embedding_dim))  # Default to zeros if missing

        else:
            # Single stock case
            tic = self.data.tic.iloc[0]
            state = (
                    [self.state[0]]
                    + [self.data.close]
                    + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                    + stock_embeddings.get(tic, [0] * self.embedding_dim)  # Add embedding
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        """
        Saves the action memory as a Pandas DataFrame while ensuring
        `date_memory` and `actions_memory` have matching lengths.
        """

        # Ensure date_list and actions_memory have the same length
        min_length = min(len(self.date_memory), len(self.actions_memory))

        # If no actions were taken, return an empty DataFrame
        if min_length == 0:
            return pd.DataFrame(columns=["date", "actions"])

        # Trim lists to match lengths
        date_list = self.date_memory[:min_length]
        action_list = self.actions_memory[:min_length]

        if len(self.df.tic.unique()) > 1:  # Multiple stocks case
            df_date = pd.DataFrame({"date": date_list})
            df_actions = pd.DataFrame(action_list)

            # Ensure correct column assignment
            if df_actions.shape[1] == len(self.data.tic.values):
                df_actions.columns = self.data.tic.values
            else:
                df_actions.columns = [f"Stock_{i}" for i in range(df_actions.shape[1])]

            df_actions.index = df_date["date"]
        else:  # Single stock case
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})

        print(f"🔹 Saved {len(df_actions)} action records.")  # Debugging output
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs



class StockTradingTransformerEnv(gym.Env):
    """A stock trading environment for Gymnasium"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # Define action and observation spaces.
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32)

        # Use iloc for integer-based indexing.
        self.data = self.df.iloc[self.day]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        self.state = self._initiate_state()
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.asset_memory = [
            self.initial_amount +
            np.sum(np.array(self.num_stock_shares) * np.array(self.state[1 : 1 + self.stock_dim]))
        ]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self._seed()

    def _initiate_state(self):
        if self.initial:
            if len(self.df.tic.unique()) > 1:
                # Multiple stocks case
                if hasattr(self.data.close, "values"):
                    close_list = self.data.close.values.tolist()
                else:
                    close_list = [self.data.close]
                tech_lists = []
                for tech in self.tech_indicator_list:
                    if hasattr(self.data[tech], "values"):
                        tech_list = self.data[tech].values.tolist()
                    else:
                        tech_list = [self.data[tech]]
                    tech_lists.append(tech_list)

                state = ([self.initial_amount] +
                         close_list +
                         self.num_stock_shares +
                         sum(tech_lists, []))

            else:
                # Single stock case
                if hasattr(self.data.close, "values"):
                    close_val = self.data.close.values[0]
                else:
                    close_val = self.data.close
                tech_values = []
                for tech in self.tech_indicator_list:
                    if hasattr(self.data[tech], "values"):
                        tech_values.append(self.data[tech].values[0])
                    else:
                        tech_values.append(self.data[tech])
                state = ([self.initial_amount] +
                         [close_val] +
                         [0] * self.stock_dim +
                         tech_values)
        else:
            if len(self.df.tic.unique()) > 1:
                if hasattr(self.data.close, "values"):
                    close_list = self.data.close.values.tolist()
                else:
                    close_list = [self.data.close]
                tech_lists = []
                for tech in self.tech_indicator_list:
                    if hasattr(self.data[tech], "values"):
                        tech_list = self.data[tech].values.tolist()
                    else:
                        tech_list = [self.data[tech]]
                    tech_lists.append(tech_list)
                state = ([self.previous_state[0]] +
                         close_list +
                         self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)] +
                         sum(tech_lists, []))
            else:
                if hasattr(self.data.close, "values"):
                    close_val = self.data.close.values[0]
                else:
                    close_val = self.data.close
                tech_values = []
                for tech in self.tech_indicator_list:
                    if hasattr(self.data[tech], "values"):
                        tech_values.append(self.data[tech].values[0])
                    else:
                        tech_values.append(self.data[tech])
                state = ([self.previous_state[0]] +
                         [close_val] +
                         self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)] +
                         tech_values)
        state = np.nan_to_num(state, nan=0.0)
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            if isinstance(self.data, pd.DataFrame):
                current_data = self.data.iloc[0]
            else:
                current_data = self.data
            if hasattr(current_data.close, "values"):
                close_list = current_data.close.values.tolist()
            else:
                close_list = [current_data.close]
            tech_lists = []
            for tech in self.tech_indicator_list:
                if hasattr(current_data[tech], "values"):
                    tech_list = current_data[tech].values.tolist()
                else:
                    tech_list = [current_data[tech]]
                tech_lists.append(tech_list)
            state = ([self.state[0]] +
                     close_list +
                     list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) +
                     sum(tech_lists, []))
        else:
            if hasattr(self.data.close, "values"):
                close_val = self.data.close.values[0]
            else:
                close_val = self.data.close
            tech_values = []
            for tech in self.tech_indicator_list:
                if hasattr(self.data[tech], "values"):
                    tech_values.append(self.data[tech].values[0])
                else:
                    tech_values.append(self.data[tech])
            state = ([self.state[0]] +
                     [close_val] +
                     list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) +
                     tech_values)
        state = np.nan_to_num(state, nan=0.0)
        return state

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            flag_index = index + 2 * self.stock_dim + 1
            print("FlagIndex: ", flag_index, "State", len(self.state))
            tradable = True
            if flag_index < len(self.state):
                tradable = (self.state[flag_index] != True)
            if tradable:
                if self.state[index + self.stock_dim + 1] > 0:
                    sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
                    sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                    self.state[0] += sell_amount
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0
            return sell_num_shares

        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    if self.state[index + self.stock_dim + 1] > 0:
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()
        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            flag_index = index + 2 * self.stock_dim + 1
            tradable = True
            if flag_index < len(self.state):
                tradable = (self.state[flag_index] != True)
            if tradable:
                price = self.state[index + 1]
                cost_pct = self.buy_cost_pct[index]
                if price == 0 or not np.isfinite(price):
                    available_amount = 0
                else:
                    available_amount = self.state[0] // (price * (1 + cost_pct))
                    if not np.isfinite(available_amount):
                        available_amount = 0
                buy_num_shares = min(available_amount, action)
                buy_amount = price * buy_num_shares * (1 + cost_pct)
                self.state[0] -= buy_amount
                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.cost += price * buy_num_shares * cost_pct
                self.trades += 1
            else:
                buy_num_shares = 0
            return buy_num_shares

        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
        return buy_num_shares

    def steptrans(self, actions):
        actions = actions * self.hmax
        actions = actions.astype(int)
        actions = np.atleast_1d(actions)
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)]) *
            np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
        )
        num_neg = int(np.sum(actions < 0))
        neg_indices = np.argsort(actions)[:num_neg]
        for index in neg_indices:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)
        num_pos = int(np.sum(actions > 0))
        pos_indices = np.argsort(actions)[::-1][:num_pos]
        for index in pos_indices:
            actions[index] = self._buy_stock(index, actions[index])
        self.day += 1

        # Check if we've exceeded the DataFrame length
        if self.day >= len(self.df):
            self.terminal = True
            self.data = self.df.iloc[-1]
            self.state = self._update_state()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)]) *
                np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )
            final_reward = 0
            return self.state, final_reward, self.terminal, False, {}

        self.data = self.df.iloc[self.day]
        self.state = self._update_state()
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)]) *
            np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
        )
        is_peak = self.data["is_peak"]
        is_valley = self.data["is_valley"]
        sell_reward = sum(actions * is_peak * self.state[1 : (self.stock_dim + 1)])
        buy_reward = sum(-actions * is_valley * self.state[1 : (self.stock_dim + 1)])
        trade_profit = end_total_asset - begin_total_asset
        trading_cost_penalty = self.cost * 0.01
        reward = trade_profit - trading_cost_penalty
        if len(self.rewards_memory) > 1:
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            sharpe_reward = np.mean(returns) / (np.std(returns) + 1e-9)
        else:
            sharpe_reward = 0
        portfolio_reward = 0.5 * reward + 0.5 * sharpe_reward
        self.reward = portfolio_reward + 0.1 * sell_reward + 0.1 * buy_reward
        self.reward = self.reward * self.reward_scaling
        self.asset_memory.append(end_total_asset)
        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)
        self.date_memory.append(self._get_date())
        self.actions_memory.append(actions.tolist())
        return self.state, self.reward, self.terminal, False, {}

    def step(self, actions):
        print(f"Day {self.day}: Actions Taken: {actions}")
        def compute_sharpe_ratio(returns):
            return np.mean(returns) / (np.std(returns) + 1e-9)
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            return self.state, self.reward, self.terminal, False, {}
        print("actions before:", actions)
        actions = actions * self.hmax  # Scale actions
        actions = actions.astype(int)  # Convert to integers (can't buy fractional shares)

        # Calculate initial portfolio value
        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )
        print("actions", actions)
        actions = np.atleast_1d(actions)
        # Execute buy/sell actions
        for index in np.argsort(actions)[:np.where(actions < 0)[0].shape[0]]:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)
        for index in np.argsort(actions)[::-1][:np.where(actions > 0)[0].shape[0]]:
            actions[index] = self._buy_stock(index, actions[index])

        # Move to next time step
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        # Calculate final portfolio value
        end_total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
        )

        #Get peak signals
        is_peak = self.data["is_peak"]
        is_valley = self.data["is_valley"]
        # Reward for selling at peaks
        sell_reward = sum(actions * is_peak * self.state[1:self.stock_dim + 1])

        # Reward for buying at valleys
        buy_reward = sum(-actions * is_valley * self.state[1:self.stock_dim + 1])

        # Calculate profit/loss per trade (instead of just total asset change)
        trade_profit = end_total_asset - begin_total_asset
        trading_cost_penalty = self.cost * 0.01  # Penalize transaction costs
        reward = trade_profit - trading_cost_penalty  # Adjusted reward

        # Normalize reward using Sharpe Ratio
        if len(self.rewards_memory) > 1:
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            sharpe_reward = compute_sharpe_ratio(returns)
        else:
            sharpe_reward = 0

        # Final reward calculation (mix of profit and risk-adjusted return)
        portfolio_reward = 0.5 * reward + 0.5 * sharpe_reward
        self.reward = portfolio_reward + 0.1 * sell_reward + 0.1 * buy_reward
        self.reward = self.reward * self.reward_scaling  # Apply scaling factor

        # Store memory for future use
        self.asset_memory.append(end_total_asset)
        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)
        self.date_memory.append(self._get_date())
        self.actions_memory.append(actions.tolist())

        return self.state, self.reward, self.terminal, False, {}

    def reset(self, *, seed=None, options=None):
        self.day = 0
        self.data = self.df.iloc[self.day]
        self.state = self._initiate_state()
        if self.initial:
            self.asset_memory = [
                self.initial_amount +
                np.sum(np.array(self.num_stock_shares) * np.array(self.state[1:(self.stock_dim + 1)]))
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)]) *
                np.array(self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )
            self.asset_memory = [previous_total_asset]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _get_date(self):
        if hasattr(self.data, "date"):
            if hasattr(self.data.date, "unique"):
                return self.data.date.unique()[0]
            else:
                return self.data.date
        else:
            return ""

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        # from stable_baselines3.common.vec_env import DummyVecEnv
        # e = DummyVecEnv([lambda: self])
        # obs, _ = self.reset()
        # return e, obs
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame({"date": date_list, "account_value": asset_list})
        return df_account_value

    def save_action_memory(self):
        """
        Saves the action memory as a Pandas DataFrame while ensuring
        date_memory and actions_memory have matching lengths.
        If there are multiple stocks, it uses ticker names (if available)
        as column headers; otherwise, it defaults to "Stock_0", "Stock_1", etc.
        """
        min_length = min(len(self.date_memory), len(self.actions_memory))
        if min_length == 0:
            return pd.DataFrame(columns=["date", "actions"])

        date_list = self.date_memory[:min_length]
        action_list = self.actions_memory[:min_length]

        # If there are multiple stocks, try to use tickers as column names.
        if len(self.df.tic.unique()) > 1:
            df_date = pd.DataFrame({"date": date_list})
            df_actions = pd.DataFrame(action_list)
            # Get unique tickers from the full DataFrame.
            tickers = self.df.tic.unique()
            # If the number of tickers matches the number of action columns, use them.
            if df_actions.shape[1] == len(tickers):
                df_actions.columns = tickers
            else:
                df_actions.columns = [f"Stock_{i}" for i in range(df_actions.shape[1])]
            df_actions.index = df_date["date"]
        else:
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})

        print(f"🔹 Saved {len(df_actions)} action records.")
        return df_actions





class PricePredictionEnvNoRandom(gym.Env):
    """
    A simplified environment for price prediction (non-random start).
    The agent’s action is a single float representing the predicted price.
    The reward is based on how close the predicted price is to the actual price.
    In this example, we use an exponential decay function:
        reward = exp( -|predicted - actual| / scale )
    so that a perfect prediction gets reward = 1, and larger errors produce rewards closer to 0.

    This environment is intended for inference/prediction. It always starts at a fixed row (start_idx)
    and then proceeds chronologically until either max_episode_length steps are reached or the data ends.
    """

    def __init__(self, df: pd.DataFrame, tech_indicator_list: list[str],
                 reward_type: str = "exp", reward_scale: float = 1.0,
                 start_idx: int = 0, max_episode_length: int = None):
        """
        Args:
            df: DataFrame containing at least 'date' and 'close' columns plus additional columns for tech indicators.
            tech_indicator_list: List of column names (features) used for constructing the observation.
            reward_type: "exp" for exponential reward, "linear" for simply negative error.
            reward_scale: Scale factor used in the exponential reward function.
            start_idx: Row index in df at which to start the episode.
            max_episode_length: Maximum number of steps in an episode (if None, runs until the end of df).
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.tech_indicator_list = tech_indicator_list
        self.reward_type = reward_type.lower()
        self.reward_scale = reward_scale
        self.start_idx = start_idx
        self.max_episode_length = max_episode_length

        # Ensure that required columns exist.
        if "date" not in self.df.columns or "close" not in self.df.columns:
            raise ValueError("DataFrame must contain 'date' and 'close' columns.")

        # Observation space: one float per tech indicator.
        obs_dim = len(tech_indicator_list)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        print("obs_dimNoRandom", obs_dim)
        # Action space: one float, the predicted price. We set a safe range here.
        # self.action_space = spaces.box.Box(low=150, high=300, shape=(1,), dtype=np.float32)
        self.action_space = spaces.box.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Internal states
        self._seed()
        self.current_step = self.start_idx
        self.terminal = False

        # Set the end index.
        self.end_idx = len(self.df) - 1
        if self.max_episode_length is not None:
            self.end_idx = min(self.end_idx, self.start_idx + self.max_episode_length - 1)

        # For logging predictions vs. actuals
        self.predictions = []
        self.actuals = []

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        """Return the current observation as an array of the tech indicator values."""
        row = self.df.loc[self.current_step]
        obs = [row[col] for col in self.tech_indicator_list]
        return np.array(obs, dtype=np.float32)

    def _get_reward(self, predicted_price: float, actual_price: float) -> float:
        """
        Compute the reward based on the prediction error.
        If reward_type is "exp", we return: exp( -|predicted - actual| / reward_scale )
        Otherwise (e.g., "linear"), we return the negative absolute error.
        """
        error = abs(predicted_price - actual_price)
        if self.reward_type == "exp":
            # reward = np.exp(-error / self.reward_scale)
            reward = 1/error if error > 0 else 10e5
        elif self.reward_type == "linear":
            reward = -error
        else:
            reward = -error
        return reward

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to a fixed start (start_idx).
        Clear prediction logs.
        """
        self.terminal = False
        self.current_step = self.start_idx
        self.predictions = []
        self.actuals = []
        return self._get_observation(), {}

    def step(self, action):
        """
        Take an action (predicted price), compute the reward, log the prediction,
        then advance one step (i.e., one day). When the episode ends, done=True.
        """
        if self.current_step > 0:
            # Previous day’s close price
            prev_close_price = float(self.df.loc[self.current_step - 1, "close"])
        else:
            prev_close_price = 415
        #predicted_price = float(action[0])
        predicted_price = (float(action[0])+1.0)/2.0 * prev_close_price

        actual_price = float(self.df.loc[self.current_step, "close"])
        self.predictions.append(predicted_price)
        self.actuals.append(actual_price)

        reward = self._get_reward(predicted_price, actual_price)
        if self.current_step > 0:
            # Previous day’s close price
            prev_close_price = float(self.df.loc[self.current_step - 1, "close"])
            # Day-to-day difference in predicted prices
            day_to_day_diff = abs(predicted_price - prev_close_price)
            # Example penalty: 0.01 * difference
            penalty_factor = 0.01
            penalty = penalty_factor * day_to_day_diff
            # Subtract penalty from reward
            reward -= penalty

        self.current_step += 1
        done = self.current_step > self.end_idx
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), done, False, {}

    def render(self, mode="human"):
        pass

    def get_sb_env(self):
        """Wrap the environment in a DummyVecEnv for Stable-Baselines3."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([lambda: self])
        obs = env.reset()
        return env, obs


import random
from stable_baselines3.common.vec_env import DummyVecEnv


from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PricePredictionEnvRandomStarts(gym.Env):
    """
    A simplified price prediction environment with random episode start dates.

    Observations: a vector of technical indicators for the current day.
    Action: a single float representing the predicted price.
    Reward: computed via an exponential reward function so that the reward is close to 1 when
            the prediction is perfect and decays as the error increases.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, tech_indicator_list: list[str],
                 max_episode_length: int = 60, reward_type: str = "exp"):
        """
        Args:
            df: DataFrame with columns including 'date', 'close', and the technical indicators.
            tech_indicator_list: List of columns (features) used for prediction.
            max_episode_length: Maximum number of days (steps) per episode.
            reward_type: 'exp' for exponential reward, 'linear' for a linear reward.
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.tech_indicator_list = tech_indicator_list
        self.max_episode_length = max_episode_length
        self.reward_type = reward_type.lower()

        # Get unique dates from the DataFrame (assumes daily data)
        self.unique_dates = sorted(self.df["date"].unique())
        self.num_days = len(self.unique_dates)

        # Observation space: one value per indicator
        obs_dim = len(tech_indicator_list)
        print("obs_dimRandom", obs_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action space: a single predicted price, assumed non-negative and bounded by a large number.
        #self.action_space = spaces.box.Box(low=150, high=300, shape=(1,), dtype=np.float32)
        self.action_space = spaces.box.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Internal states and logs
        self._seed()
        self.current_step = 0
        self.start_day_idx = 0
        self.terminal = False
        self.episode = 0

        # For logging predictions vs. actual prices
        self.predictions = []
        self.actuals = []

    def _seed(self, seed=None):
        """Ensure compatibility with NumPy's newer random API."""
        self.np_random, seed = seeding.np_random(seed)

        # ✅ Ensure np_random is using NumPy's newer Generator API
        if not isinstance(self.np_random, np.random.Generator):
            self.np_random = np.random.default_rng(seed)

        return [seed]

    def _get_observation(self):
        """Return the current observation as a numpy array of tech indicator values."""
        row = self.df.loc[self.current_step]
        obs = [row[col] for col in self.tech_indicator_list]
        return np.array(obs, dtype=np.float32)

    def _get_reward(self, predicted_price: float, actual_price: float) -> float:
        """
        Compute reward using an exponential decay function.
        When the prediction is perfect (error==0), reward=1.
        As error increases, reward decays toward 0.
        You can adjust the 'scale' parameter as a hyperparameter.
        """
        error = abs(predicted_price - actual_price)
        scale = 1.0  # Adjust this scale (e.g., set to a fraction of typical price) as needed.
        if self.reward_type == "exp":
            #reward = np.exp(-error / scale)
            reward = 1/error if error > 0 else 10e5
        elif self.reward_type == "linear":
            reward = max(0, (actual_price - error) / actual_price) if actual_price != 0 else 0
        else:
            reward = -error  # Default: negative error
        return reward

    def reset(self, *, seed=None, options=None):
        self.terminal = False
        self.episode += 1

        # Ensure max_episode_length is valid
        if self.max_episode_length > self.num_days:
            print(
                f"⚠️ max_episode_length ({self.max_episode_length}) exceeds number of unique days ({self.num_days}). Adjusting max_episode_length.")
            self.max_episode_length = self.num_days

        # Calculate maximum valid start index
        max_start = max(0, self.num_days - self.max_episode_length)

        # ✅ Use `.integers()` instead of `.randint()` to work with NumPy Generator
        self.start_day_idx = self.np_random.integers(0, max_start + 1) if max_start > 0 else 0
        start_date = self.unique_dates[self.start_day_idx]

        # Find the row in df that matches this start date
        matching_rows = self.df[self.df["date"] == start_date]
        if len(matching_rows) == 0:
            raise ValueError(f"❌ No rows found for start_date {start_date}. Check your data.")
        self.current_step = matching_rows.index[0]

        # Reset logs
        self.predictions = []
        self.actuals = []

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """
        Execute one timestep: record the prediction, compute reward, and advance one day.
        """
        if self.current_step > 0:
            # Previous day’s close price
            prev_close_price = float(self.df.loc[self.current_step - 1, "close"])
        else:
            prev_close_price = 415
        #predicted_price = float(action[0])
        predicted_price = (float(action[0]) + 1.0)/2.0 * prev_close_price

        actual_price = float(self.df.loc[self.current_step, "close"])
        self.predictions.append(predicted_price)
        self.actuals.append(actual_price)

        reward = self._get_reward(predicted_price, actual_price)
        if self.current_step > 0:
            # Previous day’s close price
            prev_close_price = float(self.df.loc[self.current_step - 1, "close"])
            # Day-to-day difference in predicted prices
            day_to_day_diff = abs(predicted_price - prev_close_price)
            # Example penalty: 0.01 * difference
            penalty_factor = 0.01
            penalty = penalty_factor * day_to_day_diff
            # Subtract penalty from reward
            reward -= penalty

        self.current_step += 1

        # Determine if the episode is done based on the number of days elapsed
        current_date = self.df.loc[self.current_step, "date"]
        day_idx = self.unique_dates.index(current_date)
        day_offset = day_idx - self.start_day_idx
        done = (day_offset >= self.max_episode_length - 1) or (self.current_step >= len(self.df) - 1)

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), done, False, {}

    def render(self, mode="human"):
        pass

    def get_sb_env(self):
        """Wrap the environment in a DummyVecEnv for use with Stable-Baselines3."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([lambda: self])
        obs = env.reset()
        return env, obs

