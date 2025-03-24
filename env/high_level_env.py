from logging import raiseExceptions
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import argparse
import os
import torch
import sys
import pathlib

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.tools.demonstration import make_q_table_reward

# Updated to match your actual dataset columns
tech_indicator_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
tech_indicator_list_trend = []  # No trend features in basic OHLCV data
clf_list = []  # No classification features

transcation_cost = 0.0002
back_time_length = 1
max_holding_number = 0.01  # Default for ETH
alpha = 0


class Testing_Env(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        clf_list=clf_list,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        initial_action=0,
    ):
        # Validate input dataframe
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns: {required_columns}")

        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.clf_list = clf_list
        self.df = df
        self.initial_action = initial_action
        self.action_space = spaces.Discrete(2)
        
        # Observation space matches the state dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(back_time_length * len(self.tech_indicator_list),)
        )
        
        self.terminal = False
        self.stack_length = back_time_length
        self.m = back_time_length
        self._update_data_window()
        
        # Initialize trading state
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        self.comission_fee = transcation_cost
        self.max_holding_number = max_holding_number
        self.reset_trading_state()

    def _update_data_window(self):
        """Update the data window to current position"""
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values if self.tech_indicator_list_trend else np.array([])
        self.clf_state = self.data[self.clf_list].values if self.clf_list else np.array([])

    def reset_trading_state(self):
        """Reset all trading-related state variables"""
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.previous_position = 0
        self.position = 0

    def calculate_value(self, price_information, position):
        """Calculate position value using Close price"""
        return price_information["Close"] * position

    def reset(self):
        """Reset the environment to initial state"""
        self.terminal = False
        self.m = self.stack_length
        self._update_data_window()
        self.reset_trading_state()
        
        # Set initial position
        self.previous_position = self.initial_action * self.max_holding_number
        self.position = self.initial_action * self.max_holding_number
        
        return self.single_state, self.trend_state, self.clf_state.reshape(-1), {
            "previous_action": self.initial_action,
        }

    def step(self, action):
        """Execute one time step in the environment"""
        # Update position
        position = self.max_holding_number * action
        self.terminal = (self.m >= len(self.df.index.unique()) - 1)
        
        # Store previous state
        previous_position = self.previous_position
        previous_price = self.data.iloc[-1]
        
        # Move window forward
        self.m += 1
        self._update_data_window()
        current_price = self.data.iloc[-1]
        
        # Calculate transaction
        self._process_transaction(previous_position, position, previous_price, current_price)
        
        # Update state
        self.previous_position = self.position
        self.previous_action = action
        
        if self.terminal:
            self._log_final_metrics(current_price)
            
        return self.single_state, self.trend_state, self.clf_state.reshape(-1), self.reward, self.terminal, {
            "previous_action": action,
        }

    def _process_transaction(self, previous_position, new_position, previous_price, current_price):
        """Handle buy/sell transactions and calculate rewards"""
        position_change = new_position - previous_position
        self.position = new_position
        
        if position_change == 0:  # No transaction
            self.reward = 0
            self.return_rate = 0
            return
            
        if position_change < 0:  # Sell
            sell_size = -position_change
            cash = sell_size * previous_price['Close'] * (1 - self.comission_fee)
            fee = self.comission_fee * sell_size * previous_price['Close']
            
            self.sell_money_memory.append(cash)
            self.needed_money_memory.append(0)
            self.comission_fee_history.append(fee)
            
            previous_value = self.calculate_value(previous_price, previous_position)
            current_value = self.calculate_value(current_price, new_position)
            
            self.reward = current_value + cash - previous_value
            self.return_rate = (current_value + cash - previous_value) / previous_value if previous_value != 0 else 0
            
        else:  # Buy
            buy_size = position_change
            needed_cash = buy_size * previous_price['Close'] * (1 + self.comission_fee)
            fee = self.comission_fee * buy_size * previous_price['Close']
            
            self.needed_money_memory.append(needed_cash)
            self.sell_money_memory.append(0)
            self.comission_fee_history.append(fee)
            
            previous_value = self.calculate_value(previous_price, previous_position)
            current_value = self.calculate_value(current_price, new_position)
            
            self.reward = current_value - needed_cash - previous_value
            self.return_rate = (current_value - needed_cash - previous_value) / (previous_value + needed_cash)
            
        self.reward_history.append(self.reward)

    def _log_final_metrics(self, current_price):
        """Calculate and log final metrics when episode terminates"""
        return_margin, pure_balance, required_money, commission_fee = self.get_final_return_rate()
        self.pured_balance = pure_balance
        self.final_balance = self.pured_balance + self.calculate_value(current_price, self.position)
        self.required_money = required_money
        print(f"Final portfolio margin: {self.final_balance / self.required_money:.4f}")

    def get_final_return_rate(self, silent=False):
        """Calculate final performance metrics"""
        sell_money = np.array(self.sell_money_memory)
        needed_money = np.array(self.needed_money_memory)
        net_cashflow = sell_money - needed_money
        final_balance = np.sum(net_cashflow)
        
        # Calculate required capital
        cumulative = np.cumsum(net_cashflow)
        required_money = -np.min(cumulative) if np.min(cumulative) < 0 else 0
        total_fees = np.sum(self.comission_fee_history)
        
        return final_balance / required_money if required_money > 0 else 0, final_balance, required_money, total_fees


class Training_Env(Testing_Env):
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        clf_list=clf_list,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        initial_action=0,
        alpha=alpha,
    ):
        super().__init__(
            df, tech_indicator_list, tech_indicator_list_trend, clf_list,
            transcation_cost, back_time_length, max_holding_number, initial_action
        )
        
        # Initialize Q-table for training
        self.q_table = make_q_table_reward(
            df,
            num_action=2,
            max_holding=max_holding_number,
            commission_fee=transcation_cost,
            reward_scale=1,
            gamma=0.99,
            max_punish=1e12
        )
        self.initial_action = initial_action

    def reset(self):
        """Reset environment and include Q-values in info"""
        single_state, trend_state, clf_state, info = super().reset()
        info['q_value'] = self.q_table[self.m - 1][self.initial_action][:]
        return single_state, trend_state, clf_state.reshape(-1), info

    def step(self, action):
        """Step environment and include Q-values in info"""
        single_state, trend_state, clf_state, reward, done, info = super().step(action)
        info['q_value'] = self.q_table[self.m - 1][action][:]
        return single_state, trend_state, clf_state.reshape(-1), reward, done, info
