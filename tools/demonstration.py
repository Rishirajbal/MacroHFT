import numpy as np
import pandas as pd

def make_q_table_reward(df, num_action=2, max_holding=0.01, reward_scale=1000, 
                       gamma=0.99, commission_fee=0.0002, max_punish=1e12):
    """
    Create a Q-table for reinforcement learning based on the given DataFrame.
    
    Args:
        df: DataFrame containing price data
        num_action: Number of possible actions
        max_holding: Maximum position size
        reward_scale: Scaling factor for rewards
        gamma: Discount factor
        commission_fee: Transaction cost
        max_punish: Maximum penalty for invalid actions
        
    Returns:
        np.ndarray: Q-table of shape (len(df), num_action, num_action)
    """
    # Validate input columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Initialize Q-table
    q_table = np.zeros((len(df), num_action, num_action))
    scale_factor = num_action - 1  # For action normalization
    
    def position_value(price_info, position):
        """Calculate dollar value of a position"""
        return price_info["Close"] * position
    
    # Build Q-table working backwards through time
    for t in range(2, len(df) + 1):
        current_price = df.iloc[-t]
        next_price = df.iloc[-t + 1]
        
        for prev_act in range(num_action):
            for curr_act in range(num_action):
                # Calculate positions
                prev_pos = (prev_act / scale_factor) * max_holding
                curr_pos = (curr_act / scale_factor) * max_holding
                pos_change = curr_pos - prev_pos
                
                # Calculate transaction costs and rewards
                if pos_change > 0:  # Buying
                    cost = pos_change * current_price['Close'] * (1 + commission_fee)
                    prev_val = position_value(current_price, prev_pos)
                    next_val = position_value(next_price, curr_pos)
                    reward = (next_val - (prev_val + cost)) * reward_scale
                else:  # Selling or holding
                    proceeds = abs(pos_change) * current_price['Close'] * (1 - commission_fee)
                    prev_val = position_value(current_price, prev_pos)
                    next_val = position_value(next_price, curr_pos)
                    reward = (next_val + proceeds - prev_val) * reward_scale
                
                # Bellman equation
                q_table[len(df) - t, prev_act, curr_act] = reward + gamma * np.max(
                    q_table[len(df) - t + 1, curr_act, :]
                )
    
    return q_table

def validate_q_table(q_table, df):
    """Validate the generated Q-table"""
    if q_table.shape != (len(df), 2, 2):
        raise ValueError(f"Q-table has incorrect shape {q_table.shape}, expected ({len(df)}, 2, 2)")
    if np.isnan(q_table).any():
        raise ValueError("Q-table contains NaN values")
    return True
