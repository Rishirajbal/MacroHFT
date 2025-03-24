import numpy as np
import pandas as pd
import gym
from gym import spaces
from pathlib import Path
import sys

# Set up paths for Google Colab compatibility
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
sys.path.insert(0, ".")

# Load technical indicators with fallback
try:
    tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
    tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()
except FileNotFoundError:
    tech_indicator_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    tech_indicator_list_trend = []

clf_list = ['slope_360', 'vol_360']
transcation_cost = 0.0002
back_time_length = 1
max_holding_number = 0.01
alpha = 0

class Testing_Env(gym.Env):
    """Environment for testing trading strategies"""
    
    def __init__(self, df, tech_indicator_list=tech_indicator_list, 
                 tech_indicator_list_trend=tech_indicator_list_trend,
                 clf_list=clf_list, transcation_cost=transcation_cost,
                 back_time_length=back_time_length, max_holding_number=max_holding_number,
                 initial_action=0):
        
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.clf_list = clf_list
        self.df = df
        self.initial_action = initial_action
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(back_time_length * len(self.tech_indicator_list),)
        
        self.terminal = False
        self.stack_length = back_time_length
        self.m = back_time_length
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        self.comission_fee = transcation_cost
        self.max_holding_number = max_holding_number
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.previous_position = 0
        self.position = 0

    def reset(self):
        """Reset the environment to initial state"""
        self.terminal = False
        self.m = back_time_length
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.clf_state = self.data[self.clf_list].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        self.position = 0
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.previous_position = self.initial_action * self.max_holding_number
        self.position = self.initial_action * self.max_holding_number
        
        return (self.single_state, 
                self.trend_state, 
                self.clf_state.reshape(-1), 
                {"previous_action": self.initial_action})

    def calculate_value(self, price_information, position):
        """Calculate the value of a position"""
        return price_information["Close"] * position

    def step(self, action):
        """Execute one step in the environment"""
        normlized_action = action
        position = self.max_holding_number * normlized_action
        self.terminal = (self.m >= len(self.df.index.unique()) - 1)
        
        previous_position = self.previous_position
        previous_price_information = self.data.iloc[-1]
        
        self.m += 1
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        current_price_information = self.data.iloc[-1]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.clf_state = self.data[self.clf_list].values
        
        self.previous_position = previous_position
        self.position = position
        self.changing = (self.position != self.previous_position)
        
        if previous_position >= position:  # Selling or holding
            self.sell_size = previous_position - position
            cash = self.sell_size * previous_price_information['Close'] * (1 - self.comission_fee)
            self.comission_fee_history.append(self.comission_fee * self.sell_size * previous_price_information['Close'])
            self.sell_money_memory.append(cash)
            self.needed_money_memory.append(0)
            
            previous_value = self.calculate_value(previous_price_information, self.previous_position)
            current_value = self.calculate_value(current_price_information, self.position)
            self.reward = current_value + cash - previous_value
            self.return_rate = (current_value + cash - previous_value) / previous_value if previous_value != 0 else 0
            self.reward_history.append(self.reward)
            
        else:  # Buying
            self.buy_size = position - previous_position
            needed_cash = self.buy_size * previous_price_information['Close'] * (1 + self.comission_fee)
            self.comission_fee_history.append(self.comission_fee * self.buy_size * previous_price_information['Close'])
            self.needed_money_memory.append(needed_cash)
            self.sell_money_memory.append(0)
            
            previous_value = self.calculate_value(previous_price_information, self.previous_position)
            current_value = self.calculate_value(current_price_information, self.position)
            self.reward = current_value - needed_cash - previous_value
            self.return_rate = (current_value - needed_cash - previous_value) / (previous_value + needed_cash)
            self.reward_history.append(self.reward)
            
        self.previous_position = self.position

        if self.terminal:
            return_margin, pure_balance, required_money, commission_fee = self.get_final_return_rate()
            self.pured_balance = pure_balance
            self.final_balance = self.pured_balance + self.calculate_value(current_price_information, self.position)
            self.required_money = required_money
            print("Portfolio margin:", self.final_balance / self.required_money)

        return (self.single_state, 
                self.trend_state, 
                self.clf_state.reshape(-1), 
                self.reward, 
                self.terminal, 
                {"previous_action": action})

    def get_final_return_rate(self, silent=False):
        """Calculate final performance metrics"""
        sell_money_memory = np.array(self.sell_money_memory)
        needed_money_memory = np.array(self.needed_money_memory)
        true_money = sell_money_memory - needed_money_memory
        final_balance = np.sum(true_money)
        
        balance_list = []
        for i in range(len(true_money)):
            balance_list.append(np.sum(true_money[:i + 1]))
            
        required_money = -np.min(balance_list)
        commission_fee = np.sum(self.comission_fee_history)
        
        return final_balance / required_money, final_balance, required_money, commission_fee

    def get_commission_fee(self):
        """Get total commission fees paid"""
        return np.sum(self.comission_fee_history)


class Training_Env(Testing_Env):
    """Environment for training trading strategies with Q-table"""
    
    def __init__(self, df, tech_indicator_list=tech_indicator_list, 
                 tech_indicator_list_trend=tech_indicator_list_trend,
                 clf_list=clf_list, transcation_cost=transcation_cost,
                 back_time_length=back_time_length, max_holding_number=max_holding_number,
                 initial_action=0, alpha=alpha):
        
        super().__init__(df, tech_indicator_list, tech_indicator_list_trend, clf_list,
                        transcation_cost, back_time_length, max_holding_number, initial_action)
        
        from MacroHFT.tools.demonstration import make_q_table_reward
        self.q_table = make_q_table_reward(
            df, num_action=2, max_holding=max_holding_number,
            commission_fee=transcation_cost, reward_scale=1,
            gamma=0.99, max_punish=1e12
        )
        self.initial_action = initial_action

    def reset(self):
        """Reset environment with Q-values"""
        single_state, trend_state, clf_state, info = super().reset()
        self.previous_action = self.initial_action
        self.previous_position = self.initial_action * self.max_holding_number
        self.position = self.initial_action * self.max_holding_number
        info['q_value'] = self.q_table[self.m - 1][self.previous_action][:]
        return single_state, trend_state, clf_state.reshape(-1), info

    def step(self, action):
        """Step with Q-values"""
        single_state, trend_state, clf_state, reward, done, info = super().step(action)
        info['q_value'] = self.q_table[self.m - 1][action][:]
        return single_state, trend_state, clf_state.reshape(-1), reward, done, info
