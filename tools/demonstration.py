import numpy as np
import pandas as pd
from pathlib import Path

def make_q_table_reward(df, num_action=2, max_holding=0.01, reward_scale=1000, 
                       gamma=0.99, commission_fee=0.0002, max_punish=1e12):
    """
    Create a Q-table optimized for the repository structure
    
    Args:
        df: DataFrame with columns matching feature lists
        num_action: Number of discrete actions (2 for long/flat)
        max_holding: Maximum position size
        reward_scale: Scaling factor for rewards
        gamma: Discount factor
        commission_fee: Transaction cost
        max_punish: Maximum penalty
        
    Returns:
        np.ndarray: Q-table of shape (len(df), num_action, num_action)
    """
    # Validate input columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")
    
    q_table = np.zeros((len(df), num_action, num_action))
    scale_factor = num_action - 1  # For action normalization
    
    for t in range(2, len(df)+1):
        current = df.iloc[-t]
        next_p = df.iloc[-t+1]
        
        for prev_act in range(num_action):
            for curr_act in range(num_action):
                prev_pos = (prev_act / scale_factor) * max_holding
                curr_pos = (curr_act / scale_factor) * max_holding
                
                if curr_pos > prev_pos:  # Buy
                    cost = (curr_pos - prev_pos) * current['Close'] * (1 + commission_fee)
                    reward = (next_p['Close']*curr_pos - cost - current['Close']*prev_pos) * reward_scale
                else:  # Sell
                    proceeds = (prev_pos - curr_pos) * current['Close'] * (1 - commission_fee)
                    reward = (next_p['Close']*curr_pos + proceeds - current['Close']*prev_pos) * reward_scale
                
                q_table[len(df)-t, prev_act, curr_act] = reward + gamma * np.max(q_table[len(df)-t+1, curr_act, :])
    
    return q_table
