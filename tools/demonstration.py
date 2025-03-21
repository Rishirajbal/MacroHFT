import numpy as np
import pandas as pd


def make_q_table_reward(df: pd.DataFrame,
                        num_action,
                        max_holding,
                        reward_scale=1000,
                        gamma=0.999,
                        commission_fee=0.001,
                        max_punish=1e12):
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
    # Ensure the DataFrame has the required columns
    required_columns = ['Close']  # Add other required columns if needed
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"The DataFrame is missing one or more required columns: {required_columns}")

    # Initialize the Q-table with zeros
    q_table = np.zeros((len(df), num_action, num_action))

    def calculate_value(price_information, position):
        """
        Calculate the value of a position based on the price information.

        Args:
            price_information (pd.Series): Price information for a specific time step.
            position (float): The current position.

        Returns:
            float: The value of the position.
        """
        return price_information["Close"] * position  # Use 'Close' instead of 'close'

    scale_factor = num_action - 1

    # Iterate over the DataFrame to populate the Q-table
    for t in range(2, len(df) + 1):
        current_price_information = df.iloc[-t]
        future_price_information = df.iloc[-t + 1]
        for previous_action in range(num_action):
            for current_action in range(num_action):
                if current_action > previous_action:
                    # Calculate position changes for buying
                    previous_position = previous_action / scale_factor * max_holding
                    current_position = current_action / scale_factor * max_holding
                    position_change = (current_action - previous_action) / scale_factor * max_holding

                    # Calculate the cost of buying
                    buy_money = position_change * current_price_information['Close'] * (1 + commission_fee)  # Use 'Close'

                    # Calculate current and future values
                    current_value = calculate_value(current_price_information, previous_position)
                    future_value = calculate_value(future_price_information, current_position)

                    # Calculate the reward
                    reward = future_value - (current_value + buy_money)
                    reward = reward_scale * reward

                    # Update the Q-table
                    q_table[len(df) - t][previous_action][current_action] = reward + gamma * np.max(
                        q_table[len(df) - t + 1][current_action][:])
                else:
                    # Calculate position changes for selling
                    previous_position = previous_action / scale_factor * max_holding
                    current_position = current_action / scale_factor * max_holding
                    position_change = (previous_action - current_action) / scale_factor * max_holding

                    # Calculate the revenue from selling
                    sell_money = position_change * current_price_information['Close'] * (1 - commission_fee)  # Use 'Close'

                    # Calculate current and future values
                    current_value = calculate_value(current_price_information, previous_position)
                    future_value = calculate_value(future_price_information, current_position)

                    # Calculate the reward
                    reward = future_value + sell_money - current_value
                    reward = reward_scale * reward

                    # Update the Q-table
                    q_table[len(df) - t][previous_action][current_action] = reward + gamma * np.max(
                        q_table[len(df) - t + 1][current_action][:])

    return q_table
