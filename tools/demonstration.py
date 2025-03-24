import numpy as np
import pandas as pd


def make_q_table_reward(df: pd.DataFrame,
                        num_action: int,
                        max_holding: float,
                        reward_scale: float = 1000,
                        gamma: float = 0.99,
                        commission_fee: float = 0.0002,  # Matches transcation_cost from other files
                        max_punish: float = 1e12) -> np.ndarray:
    """
    Create a Q-table for reinforcement learning based on the given DataFrame.
    Updated to align with high_level.py and high_level_env.py configurations.

    Args:
        df: DataFrame containing OHLCV price information
        num_action: Number of discrete actions (2 for long/flat)
        max_holding: Maximum position size (from high_level_env.py)
        reward_scale: Scaling factor for rewards
        gamma: Discount factor for future rewards
        commission_fee: Transaction cost (matches high_level_env.py)
        max_punish: Maximum penalty for invalid actions

    Returns:
        Q-table of shape (len(df), num_action, num_action)
    """
    # Validate required columns - matches your dataset
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    # Initialize Q-table with proper dimensions
    q_table = np.zeros((len(df), num_action, num_action))
    scale_factor = num_action - 1  # Normalization factor for actions

    def calculate_position_value(price_info: pd.Series, position: float) -> float:
        """Calculate dollar value of a position using Close price"""
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
                
                # Calculate transaction costs
                if pos_change > 0:  # Buy
                    cost = pos_change * current_price['Close'] * (1 + commission_fee)
                    prev_value = calculate_position_value(current_price, prev_pos)
                    next_value = calculate_position_value(next_price, curr_pos)
                    reward = (next_value - (prev_value + cost)) * reward_scale
                else:  # Sell or hold
                    proceeds = abs(pos_change) * current_price['Close'] * (1 - commission_fee)
                    prev_value = calculate_position_value(current_price, prev_pos)
                    next_value = calculate_position_value(next_price, curr_pos)
                    reward = (next_value + proceeds - prev_value) * reward_scale
                
                # Bellman equation
                q_table[len(df) - t, prev_act, curr_act] = reward + gamma * np.max(
                    q_table[len(df) - t + 1, curr_act, :]
                )

    return q_table


def validate_q_table(q_table: np.ndarray, df: pd.DataFrame) -> bool:
    """Sanity check for the generated Q-table"""
    if q_table.shape != (len(df), 2, 2):
        raise ValueError(f"Q-table has incorrect shape {q_table.shape}, expected ({len(df)}, 2, 2)")
    
    if np.isnan(q_table).any():
        raise ValueError("Q-table contains NaN values")
        
    return True
