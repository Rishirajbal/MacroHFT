import numpy as np
import pandas as pd
import gym
from gym import spaces
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Set up paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

# Load technical indicators from repo
try:
    tech_indicator_list = np.load(REPO_ROOT/'data/feature_list/single_features.npy', allow_pickle=True).tolist()
    tech_indicator_list_trend = np.load(REPO_ROOT/'data/feature_list/trend_features.npy', allow_pickle=True).tolist()
except FileNotFoundError:
    # Fallback to basic indicators
    tech_indicator_list = ['Open', 'High', 'Low', 'Close', 'Volume']
    tech_indicator_list_trend = ['Adj Close']
    print("Using fallback technical indicators")

clf_list = ['slope_360', 'vol_360']
transcation_cost = 0.0002
back_time_length = 1
max_holding_number = 0.01
alpha = 0

class Testing_Env(gym.Env):
    def __init__(self, df, tech_indicator_list=tech_indicator_list, 
                 tech_indicator_list_trend=tech_indicator_list_trend,
                 clf_list=clf_list, transcation_cost=transcation_cost,
                 back_time_length=back_time_length, max_holding_number=max_holding_number,
                 initial_action=0):
        
        self.df = df.reset_index(drop=True)
        self._validate_columns(tech_indicator_list + tech_indicator_list_trend + clf_list + ['Close'])
        
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.clf_list = clf_list
        self.transcation_cost = transcation_cost
        self.back_time_length = back_time_length
        self.max_holding_number = max_holding_number
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(back_time_length * len(tech_indicator_list),)
        )
        self.reset()

    def _validate_columns(self, required_columns):
        missing = set(required_columns) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def reset(self):
        self.current_step = self.back_time_length
        self.position = 0
        self.portfolio_value = 10000  # Initial capital
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        obs = self.df.loc[
            self.current_step-self.back_time_length:self.current_step,
            self.tech_indicator_list
        ].values
        return obs

    def step(self, action):
        if self.done:
            raise ValueError("Episode has completed")
            
        prev_price = self.df.loc[self.current_step-1, 'Close']
        curr_price = self.df.loc[self.current_step, 'Close']
        
        # Execute trade
        if action == 1:  # Buy
            cost = self.max_holding_number * prev_price * (1 + self.transcation_cost)
            self.portfolio_value -= cost
            self.position = self.max_holding_number
        else:  # Sell
            proceeds = self.position * prev_price * (1 - self.transcation_cost)
            self.portfolio_value += proceeds
            self.position = 0
            
        # Calculate reward
        new_value = self.portfolio_value + (self.position * curr_price)
        reward = np.log(new_value / (self.portfolio_value + (self.position * prev_price)))
        
        self.current_step += 1
        self.done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, self.done, {}

class Training_Env(Testing_Env):
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

    def reset(self):
        state = super().reset()
        info = {'q_value': self.q_table[self.current_step-1][self.position > 0][:]}
        return (*state[:-1], info)

    def step(self, action):
        state, reward, done, _ = super().step(action)
        info = {'q_value': self.q_table[self.current_step-1][action][:]}
        return (*state, reward, done, info)
