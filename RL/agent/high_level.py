import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# Suppress TensorFlow/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ====================== Configuration ======================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    lr = 0.0001
    batch_size = 32
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update = 10
    memory_size = 10000
    num_episodes = 150
    
    # Data paths
    train_path = "/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_train.csv"
    val_path = "/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_validate.csv"
    test_path = "/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_test.csv"
    
    # Dataset limitation (set to None for full dataset)
    max_train_rows = 10000  # Reduced dataset size for testing
    max_val_rows = 2000
    
    tech_indicators = ['Open', 'High', 'Low', 'Close', 'Volume']

# Set random seeds
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)
random.seed(Config.seed)

# ====================== Neural Network ======================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ====================== Trading Environment ======================
class TradingEnv:
    def __init__(self, df, initial_balance=10000, transaction_cost=0.0002):
        print("\n" + "="*50)
        print("Debug: Initializing environment...")
        print(f"Raw data shape: {df.shape}")
        
        # Data preprocessing
        self.df = df[Config.tech_indicators].copy()
        self.df = self.df.apply(pd.to_numeric, errors='coerce').dropna()
        print(f"Clean data shape: {self.df.shape}")
        print("Sample data points:")
        print(self.df.head(2))
        
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.action_space = [0, 1]
        self.reset()
        print("Environment initialized successfully!")
        print("="*50 + "\n")
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holding = 0
        self.portfolio_value = [self.initial_balance]
        return self._get_state()
    
    def _get_state(self):
        state = self.df.iloc[self.current_step].values.astype(np.float32)
        return torch.FloatTensor(state).to(Config.device)
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_state(), 0, True, {}
            
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close']
        
        # Execute action
        if action == 1 and self.holding == 0:
            cost = current_price * (1 + self.transaction_cost)
            self.holding = self.balance / cost
            self.balance = 0
        
        elif action == 0 and self.holding > 0:
            self.balance = self.holding * current_price * (1 - self.transaction_cost)
            self.holding = 0
        
        # Update portfolio
        new_value = self.balance + (self.holding * next_price)
        reward = np.log(new_value / self.portfolio_value[-1]) if self.portfolio_value[-1] > 0 else 0
        self.portfolio_value.append(new_value)
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._get_state(), reward, done, {}

# ====================== RL Agent ======================
class DQNAgent:
    def __init__(self, input_dim, output_dim):
        print("Initializing DQN agent...")
        self.policy_net = DQN(input_dim, output_dim).to(Config.device)
        self.target_net = DQN(input_dim, output_dim).to(Config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.lr)
        self.memory = deque(maxlen=Config.memory_size)
        self.epsilon = Config.epsilon_start
        self.loss_fn = nn.SmoothL1Loss()
        print(f"Agent initialized on {Config.device}")
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        with torch.no_grad():
            return self.policy_net(state).argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_model(self):
        if len(self.memory) < Config.batch_size:
            return 0
        
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
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        self.epsilon = max(Config.epsilon_end, 
                         self.epsilon * Config.epsilon_decay)
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ====================== Training Loop ======================
def train():
    print("="*50)
    print("Starting training process...")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    # Load and limit data
    print("Loading training data...")
    try:
        train_df = pd.read_csv(Config.train_path)
        val_df = pd.read_csv(Config.val_path)
        
        # Apply dataset limitation
        if Config.max_train_rows:
            train_df = train_df.iloc[:Config.max_train_rows]
        if Config.max_val_rows:
            val_df = val_df.iloc[:Config.max_val_rows]
            
        print(f"Using LIMITED dataset: Train {len(train_df)} rows, Val {len(val_df)} rows")
        print("\nTraining data sample:")
        print(train_df.head(2))
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    # Initialize components
    print("\nInitializing environment and agent...")
    env = TradingEnv(train_df)
    agent = DQNAgent(input_dim=len(Config.tech_indicators), output_dim=2)
    
    best_val_return = -np.inf
    returns = []
    losses = []
    
    print("\n" + "="*50)
    print("Beginning training loop...")
    print(f"Total episodes: {Config.num_episodes}")
    print("="*50 + "\n")
    
    for episode in range(Config.num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_loss = 0
        update_count = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.update_model()
            if loss > 0:
                episode_loss += loss
                update_count += 1
            
            state = next_state
            total_reward += reward
        
        avg_loss = episode_loss / update_count if update_count > 0 else 0
        
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
        
        returns.append(val_return)
        losses.append(avg_loss)
        
        if val_return > best_val_return:
            best_val_return = val_return
            torch.save(agent.policy_net.state_dict(), "best_model.pth")
        
        if episode % Config.target_update == 0:
            agent.update_target()
        
        print(f"Episode {episode+1:03d}/{Config.num_episodes} | "
              f"Train: {total_reward:+.2f} | "
              f"Val: {val_return:+.2f} | "
              f"ε: {agent.epsilon:.2f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Steps: {env.current_step}")
    
    # Training complete
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation return: {best_val_return:.2f}")
    print(f"Final epsilon: {agent.epsilon:.2f}")
    print("="*50 + "\n")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(returns)
    plt.title("Validation Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid()
    
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()

if __name__ == "__main__":
    train()
