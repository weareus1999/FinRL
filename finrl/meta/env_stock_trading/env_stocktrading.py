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


import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
from ta.volatility import BollingerBands

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






