import numpy as np
import pandas as pd


def make_q_table_reward(df: pd.DataFrame,
                        num_action: int,
                        max_holding: float,
                        reward_scale: float = 1000,
                        gamma: float = 0.999,
                        commission_fee: float = 0.001,
                        max_punish: float = 1e12):
    """
    Create a Q-table for reinforcement learning based on the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing price information.
        num_action (int): The number of possible actions.
        max_holding (float): The maximum holding position.
        reward_scale (float): Scaling factor for rewards.
        gamma (float): Discount factor for future rewards.
        commission_fee (float): Commission fee for transactions.
        max_punish (float): Maximum punishment for invalid actions.

    Returns:
        np.ndarray: A Q-table of shape (len(df), num_action, num_action).
    """
    # Validate input DataFrame
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"The DataFrame is missing one or more required columns: {required_columns}")

    # Initialize the Q-table
    q_table = np.zeros((len(df), num_action, num_action))

    def calculate_position_value(price: float, position: float) -> float:
        """Calculate the value of a position based on price."""
        return price * position

    scale_factor = num_action - 1

    # Iterate over the DataFrame to populate the Q-table
    for t in range(2, len(df) + 1):
        current_price_info = df.iloc[-t]
        future_price_info = df.iloc[-t + 1]
        
        for prev_action in range(num_action):
            for curr_action in range(num_action):
                prev_position = prev_action / scale_factor * max_holding
                curr_position = curr_action / scale_factor * max_holding
                position_change = (curr_action - prev_action) / scale_factor * max_holding

                if curr_action > prev_action:
                    # Buying scenario
                    transaction_cost = position_change * current_price_info['Close'] * (1 + commission_fee)
                    current_value = calculate_position_value(current_price_info['Close'], prev_position)
                    future_value = calculate_position_value(future_price_info['Close'], curr_position)
                    reward = future_value - (current_value + transaction_cost)
                elif curr_action < prev_action:
                    # Selling scenario
                    transaction_cost = position_change * current_price_info['Close'] * (1 - commission_fee)
                    current_value = calculate_position_value(current_price_info['Close'], prev_position)
                    future_value = calculate_position_value(future_price_info['Close'], curr_position)
                    reward = future_value + transaction_cost - current_value
                else:
                    # Holding scenario
                    reward = 0

                # Scale and penalize invalid actions
                reward *= reward_scale
                if curr_position < 0 or curr_position > max_holding:
                    reward -= max_punish

                # Update Q-table with discounted future rewards
                q_table[len(df) - t][prev_action][curr_action] = reward + gamma * np.max(
                    q_table[len(df) - t + 1][curr_action][:])

    return q_table
