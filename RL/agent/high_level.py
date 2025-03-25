import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow/CUDA warnings if you're not using TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ====================== Configuration ======================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    lr = 0.001
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update = 10
    memory_size = 10000
    num_episodes = 1000
    
    # Your data paths
    train_path = "/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_train.csv"
    val_path = "/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_validate.csv"
    test_path = "/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_test.csv"

# ====================== Neural Networks ======================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# ====================== Trading Environment ======================
class TradingEnv:
    def __init__(self, df, tech_indicators=['Close'], initial_balance=10000, transaction_cost=0.0002):
        self.df = df.reset_index(drop=True)
        self.tech_indicators = tech_indicators
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.action_space = [0, 1]  # 0: hold, 1: buy
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holding = 0
        self.portfolio_value = [self.initial_balance]
        return self._get_state()
    
    def _get_state(self):
        state = self.df.loc[self.current_step, self.tech_indicators].values
        return torch.FloatTensor(state).to(Config.device)
    
    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        next_price = self.df.loc[self.current_step + 1, 'Close'] if self.current_step + 1 < len(self.df) else current_price
        
        # Execute action
        if action == 1 and self.holding == 0:  # Buy
            cost = current_price * (1 + self.transaction_cost)
            self.holding = self.balance / cost
            self.balance = 0
        
        elif action == 0 and self.holding > 0:  # Sell
            self.balance = self.holding * current_price * (1 - self.transaction_cost)
            self.holding = 0
        
        # Update portfolio value
        new_value = self.balance + (self.holding * next_price)
        reward = np.log(new_value / self.portfolio_value[-1])
        self.portfolio_value.append(new_value)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._get_state(), reward, done, {}

# ====================== RL Agent ======================
class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.policy_net = DQN(input_dim, output_dim).to(Config.device)
        self.target_net = DQN(input_dim, output_dim).to(Config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.lr)
        self.memory = deque(maxlen=Config.memory_size)
        self.epsilon = Config.epsilon_start
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        with torch.no_grad():
            return self.policy_net(state).argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_model(self):
        if len(self.memory) < Config.batch_size:
            return
        
        batch = random.sample(self.memory, Config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.LongTensor(actions).to(Config.device)
        rewards = torch.FloatTensor(rewards).to(Config.device)
        dones = torch.FloatTensor(dones).to(Config.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * Config.gamma * next_q
        
        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(Config.epsilon_end, 
                          self.epsilon * Config.epsilon_decay)
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ====================== Training Loop ======================
def train():
    # Load data
    train_df = pd.read_csv(Config.train_path)
    val_df = pd.read_csv(Config.val_path)
    
    # Initialize
    env = TradingEnv(train_df)
    agent = DQNAgent(input_dim=len(env.tech_indicators), output_dim=2)
    
    best_val_return = -np.inf
    
    for episode in range(Config.num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update_model()
            state = next_state
            total_reward += reward
        
        # Validation
        val_env = TradingEnv(val_df)
        val_state = val_env.reset()
        val_done = False
        val_return = 0
        
        with torch.no_grad():
            while not val_done:
                val_action = agent.policy_net(val_state).argmax().item()
                val_state, val_reward, val_done, _ = val_env.step(val_action)
                val_return += val_reward
        
        # Save best model
        if val_return > best_val_return:
            best_val_return = val_return
            torch.save(agent.policy_net.state_dict(), "best_model.pth")
        
        # Update target network
        if episode % Config.target_update == 0:
            agent.update_target()
        
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Val Return: {val_return:.2f}, Epsilon: {agent.epsilon:.2f}")

if __name__ == "__main__":
    train()
