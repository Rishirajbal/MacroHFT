import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up paths for Google Colab
ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

# Import custom modules
from MacroHFT.model.net import subagent, hyperagent
from MacroHFT.env.high_level_env import Testing_Env, Training_Env
from MacroHFT.RL.util.replay_buffer import ReplayBuffer_High
from MacroHFT.RL.util.memory import episodicmemory

# Set environment variables for threading
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

class LinearDecaySchedule:
    """Linear epsilon decay schedule for exploration"""
    def __init__(self, start_epsilon, end_epsilon, decay_length):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_length = decay_length

    def get_epsilon(self, epoch):
        if epoch >= self.decay_length:
            return self.end_epsilon
        return self.start_epsilon - (self.start_epsilon - self.end_epsilon) * (epoch / self.decay_length)

class DQNAgent:
    """High-level trading agent using DQN with multiple sub-agents"""
    
    def __init__(self, args):
        self.args = args
        self._setup_device()
        self._setup_paths()
        self._load_technical_indicators()
        self._init_parameters()
        self._init_networks()
        self._setup_training()
        self._setup_logging()
        
    def _setup_device(self):
        """Configure device and random seeds for reproducibility"""
        self.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        
    def _setup_paths(self):
        """Create directories for results and models"""
        self.result_path = Path("./result/high_level") / self.args.dataset / self.args.exp
        self.model_path = self.result_path / f"seed_{self.args.seed}"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Data paths (Google Colab compatible)
        base_data_path = Path('/content/drive/MyDrive/MacroHFT/data') / self.args.dataset / 'whole'
        self.train_data_path = base_data_path / 'df_train.csv'
        self.val_data_path = base_data_path / 'df_validate.csv'
        self.test_data_path = base_data_path / 'df_test.csv'
        
    def _load_technical_indicators(self):
        """Load technical indicators with fallback"""
        try:
            self.tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
            self.tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()
        except FileNotFoundError:
            self.tech_indicator_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            self.tech_indicator_list_trend = []
        self.clf_list = ['slope_360', 'vol_360']
            
    def _init_parameters(self):
        """Initialize agent parameters"""
        self.n_action = 2  # Binary action space
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        
        # Set max holding based on dataset
        if "BTC" in self.args.dataset:
            self.max_holding_number = 0.01
        elif "ETH" in self.args.dataset:
            self.max_holding_number = 0.2
        elif "DOT" in self.args.dataset or "LTC" in self.args.dataset:
            self.max_holding_number = 10
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
            
    def _init_networks(self):
        """Initialize neural networks"""
        # Subagents for different market conditions
        self.slope_agents = nn.ModuleList([
            subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device) 
            for _ in range(3)
        ])
        self.vol_agents = nn.ModuleList([
            subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
            for _ in range(3)
        ])
        
        # Load pretrained subagents if available
        self._load_pretrained_subagents()
        
        # Hyperagent that combines subagents
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
        
    def _load_pretrained_subagents(self):
        """Load pretrained subagent models"""
        model_base_path = Path("./result/low_level") / self.args.dataset / "best_model"
        
        for i in range(3):
            slope_path = model_base_path / "slope" / str(i+1) / "best_model.pkl"
            vol_path = model_base_path / "vol" / str(i+1) / "best_model.pkl"
            
            if slope_path.exists():
                self.slope_agents[i].load_state_dict(torch.load(slope_path, map_location=self.device))
                self.slope_agents[i].eval()
            if vol_path.exists():
                self.vol_agents[i].load_state_dict(torch.load(vol_path, map_location=self.device))
                self.vol_agents[i].eval()
                
    def _setup_training(self):
        """Configure training components"""
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(), lr=self.args.lr)
        self.loss_func = nn.MSELoss()
        self.epsilon_scheduler = LinearDecaySchedule(
            self.args.epsilon_start, self.args.epsilon_end, self.args.decay_length)
        self.epsilon = self.args.epsilon_start
        
        # Memory and replay buffer
        self.memory = episodicmemory(4320, 5, self.n_state_1, self.n_state_2, 64, self.device)
        self.replay_buffer = ReplayBuffer_High(self.args, self.n_state_1, self.n_state_2, self.n_action)
        
        self.update_counter = 0
        
    def _setup_logging(self):
        """Initialize logging and tensorboard"""
        self.log_path = self.model_path / "log"
        self.log_path.mkdir(exist_ok=True)
        self.writer = SummaryWriter(str(self.log_path))
        
    def calculate_q(self, w, qs):
        """Combine Q-values from subagents using hyperagent weights"""
        q_tensor = torch.stack(qs).permute(1, 0, 2)
        weights_reshaped = w.view(-1, 1, 6)  # 6 subagents (3 slope + 3 vol)
        return torch.bmm(weights_reshaped, q_tensor).squeeze(1)
        
    def update(self, replay_buffer):
        """Update networks using experience replay"""
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Handle empty state_trend case
        if self.n_state_2 == 0:
            batch['state_trend'] = torch.zeros_like(batch['state'][:, :0])
            batch['next_state_trend'] = torch.zeros_like(batch['next_state'][:, :0])

        # Get current and target Q-values
        w_current = self.hyperagent(batch['state'], batch['state_trend'], batch['state_clf'], batch['previous_action'])
        w_next = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])
        w_next_ = self.hyperagent(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])

        # Get Q-values from all subagents
        qs_current = [agent(batch['state'], batch['state_trend'], batch['previous_action']) for agent in self.slope_agents + self.vol_agents]
        qs_next = [agent(batch['next_state'], batch['next_state_trend'], batch['next_previous_action']) for agent in self.slope_agents + self.vol_agents]

        # Calculate combined Q-values
        q_distribution = self.calculate_q(w_current, qs_current)
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)
        
        # Double DQN target calculation
        a_argmax = self.calculate_q(w_next_, qs_next).argmax(dim=-1, keepdim=True)
        q_nexts = self.calculate_q(w_next, qs_next)
        q_target = batch['reward'] + self.args.gamma * (1 - batch['terminal']) * q_nexts.gather(-1, a_argmax).squeeze(-1)

        # Calculate losses
        td_error = self.loss_func(q_current, q_target)
        memory_error = self.loss_func(q_current, batch['q_memory'])
        
        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )

        # Combined loss with regularization terms
        loss = td_error + self.args.alpha * memory_error + self.args.beta * KL_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hyperagent.parameters(), 1)
        self.optimizer.step()
        
        # Update target network
        for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            
        self.update_counter += 1
        return td_error.item(), memory_error.item(), KL_loss.item(), torch.mean(q_current).item(), torch.mean(q_target).item()

    def act(self, state, state_trend, state_clf, info):
        """Select action using epsilon-greedy policy"""
        # Convert inputs to tensors
        x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device) if len(state_trend) > 0 else torch.zeros(1, 0).to(self.device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device) if len(state_clf) > 0 else torch.zeros(1, 0).to(self.device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0)
        
        # Epsilon-greedy action selection
        if np.random.uniform() < (1 - self.epsilon):
            with torch.no_grad():
                qs = [agent(x1, x2, previous_action) for agent in self.slope_agents + self.vol_agents]
                w = self.hyperagent(x1, x2, x3, previous_action)
                actions_value = self.calculate_q(w, qs)
                action = torch.max(actions_value, 1)[1].item()
        else:
            action = random.choice([0, 1])
        return action

    def train(self):
        """Main training loop"""
        best_return_rate = -float('inf')
        best_model = None
        
        for epoch in range(1, self.args.epoch_number + 1):
            print(f'\nEpoch {epoch}/{self.args.epoch_number}')
            
            # Initialize training environment
            df_train = pd.read_csv(self.train_data_path)
            train_env = Training_Env(
                df=df_train,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.args.transcation_cost,
                back_time_length=self.args.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=random.choice([0, 1]),
                alpha=0
            )
            
            # Training episode
            state, state_trend, state_clf, info = train_env.reset()
            episode_reward = 0
            step_counter = 0
            
            while True:
                action = self.act(state, state_trend, state_clf, info)
                next_state, next_state_trend, next_state_clf, reward, done, next_info = train_env.step(action)
                
                # Store transition in replay buffer
                self._store_transition(
                    state, state_trend, state_clf, info, action, reward,
                    next_state, next_state_trend, next_state_clf, next_info, done
                )
                
                episode_reward += reward
                state, state_trend, state_clf, info = next_state, next_state_trend, next_state_clf, next_info
                step_counter += 1
                
                # Update networks periodically
                if len(self.replay_buffer) > self.args.batch_size and step_counter % self.args.eval_update_freq == 0:
                    self._update_networks()
                
                if done:
                    break
            
            # Log training metrics
            final_balance = train_env.final_balance
            required_money = train_env.required_money
            return_rate = final_balance / required_money
            self.writer.add_scalar("Train/ReturnRate", return_rate, epoch)
            self.writer.add_scalar("Train/FinalBalance", final_balance, epoch)
            self.writer.add_scalar("Train/RequiredMoney", required_money, epoch)
            self.writer.add_scalar("Train/RewardSum", episode_reward, epoch)
            
            # Run validation
            val_return = self._validate(epoch)
            if val_return > best_return_rate:
                best_return_rate = val_return
                best_model = self.hyperagent.state_dict()
                torch.save(best_model, self.model_path / "best_model.pkl")
                print(f"New best model saved with validation return: {best_return_rate:.4f}")
                
            # Update exploration rate
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch)
        
        # Final testing with best model
        print("\nTraining completed. Running final test...")
        self._test(best_model)
        
    def _store_transition(self, state, state_trend, state_clf, info, action, reward,
                         next_state, next_state_trend, next_state_clf, next_info, done):
        """Store experience in replay buffer and memory"""
        hs = self._calculate_hidden(state, state_trend, info)
        q = reward + self.args.gamma * (1 - done) * self.q_estimate(next_state, next_state_trend, next_state_clf, next_info)
        q_memory = self.memory.query(hs, action)
        
        if np.isnan(q_memory):
            q_memory = q
            
        self.replay_buffer.store_transition(
            state, state_trend, state_clf, info['previous_action'], info['q_value'],
            action, reward, next_state, next_state_trend, next_state_clf,
            next_info['previous_action'], next_info['q_value'], done, q_memory
        )
        self.memory.add(hs, action, q, state, state_trend, info['previous_action'])
        
    def _update_networks(self):
        """Perform network updates and logging"""
        for _ in range(self.args.update_times):
            td_error, memory_error, kl_loss, q_eval, q_target = self.update(self.replay_buffer)
            
            if self.update_counter % self.args.q_value_memorize_freq == 1:
                self.writer.add_scalar("Loss/TDError", td_error, self.update_counter)
                self.writer.add_scalar("Loss/MemoryError", memory_error, self.update_counter)
                self.writer.add_scalar("Loss/KLLoss", kl_loss, self.update_counter)
                self.writer.add_scalar("QValues/Eval", q_eval, self.update_counter)
                self.writer.add_scalar("QValues/Target", q_target, self.update_counter)
                
        # Periodically re-encode memory
        if len(self.replay_buffer) > 4320:
            self.memory.re_encode(self.hyperagent)

    def _validate(self, epoch):
        """Run validation episode"""
        self.hyperagent.eval()
        df_val = pd.read_csv(self.val_data_path)
        val_env = Testing_Env(
            df=df_val,
            tech_indicator_list=self.tech_indicator_list,
            tech_indicator_list_trend=self.tech_indicator_list_trend,
            clf_list=self.clf_list,
            transcation_cost=self.args.transcation_cost,
            back_time_length=self.args.back_time_length,
            max_holding_number=self.max_holding_number,
            initial_action=0
        )
        
        state, state_trend, state_clf, info = val_env.reset()
        while True:
            action = self.act_test(state, state_trend, state_clf, info)
            next_state, next_state_trend, next_state_clf, reward, done, next_info = val_env.step(action)
            state, state_trend, state_clf, info = next_state, next_state_trend, next_state_clf, next_info
            if done:
                break
        
        return_rate = val_env.final_balance / val_env.required_money
        self.writer.add_scalar("Validation/ReturnRate", return_rate, epoch)
        return return_rate
        
    def _test(self, model_state_dict):
        """Run final test with best model"""
        self.hyperagent.load_state_dict(model_state_dict)
        self.hyperagent.eval()
        
        df_test = pd.read_csv(self.test_data_path)
        test_env = Testing_Env(
            df=df_test,
            tech_indicator_list=self.tech_indicator_list,
            tech_indicator_list_trend=self.tech_indicator_list_trend,
            clf_list=self.clf_list,
            transcation_cost=self.args.transcation_cost,
            back_time_length=self.args.back_time_length,
            max_holding_number=self.max_holding_number,
            initial_action=0
        )
        
        # Run test episode
        state, state_trend, state_clf, info = test_env.reset()
        action_history = []
        reward_history = []
        
        while True:
            action = self.act_test(state, state_trend, state_clf, info)
            next_state, next_state_trend, next_state_clf, reward, done, next_info = test_env.step(action)
            
            action_history.append(action)
            reward_history.append(reward)
            
            state, state_trend, state_clf, info = next_state, next_state_trend, next_state_clf, next_info
            if done:
                break
        
        # Save test results
        final_balance = test_env.final_balance
        required_money = test_env.required_money
        commission_fee = test_env.get_commission_fee()
        
        results = {
            'actions': np.array(action_history),
            'rewards': np.array(reward_history),
            'final_balance': np.array([final_balance]),
            'required_money': np.array([required_money]),
            'commission_fee': np.array([commission_fee])
        }
        
        # Save results to files
        for name, data in results.items():
            np.save(self.model_path / f"{name}.npy", data)
            
        print(f"\nTest Results:")
        print(f"Final Balance: {final_balance:.2f}")
        print(f"Required Money: {required_money:.2f}")
        print(f"Return Rate: {final_balance / required_money:.4f}")
        print(f"Commission Fees: {commission_fee:.2f}")
        
    def act_test(self, state, state_trend, state_clf, info):
        """Select action without exploration (for testing)"""
        with torch.no_grad():
            return self.act(state, state_trend, state_clf, info)
            
    def q_estimate(self, state, state_trend, state_clf, info):
        """Estimate Q-value for given state"""
        with torch.no_grad():
            x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device) if len(state_trend) > 0 else torch.zeros(1, 0).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device) if len(state_clf) > 0 else torch.zeros(1, 0).to(self.device)
            previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0)
            
            qs = [agent(x1, x2, previous_action) for agent in self.slope_agents + self.vol_agents]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            return torch.max(actions_value, 1)[0].item()
            
    def _calculate_hidden(self, state, state_trend, info):
        """Calculate hidden state representation"""
        with torch.no_grad():
            x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device) if len(state_trend) > 0 else torch.zeros(1, 0).to(self.device)
            previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0)
            return self.hyperagent.encode(x1, x2, previous_action).cpu().numpy()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="High-Level Trading Agent")
    
    # Training parameters
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--update_times", type=int, default=10, help="Number of updates per step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    
    # Exploration parameters
    parser.add_argument("--epsilon_start", type=float, default=0.7, help="Initial exploration rate")
    parser.add_argument("--epsilon_end", type=float, default=0.3, help="Final exploration rate")
    parser.add_argument("--decay_length", type=int, default=5, help="Epsilon decay length in epochs")
    
    # Environment parameters
    parser.add_argument("--transcation_cost", type=float, default=0.0002, help="Transaction cost percentage")
    parser.add_argument("--back_time_length", type=int, default=1, help="Lookback window size")
    parser.add_argument("--n_step", type=int, default=1, help="N-step returns")
    
    # Logging and evaluation
    parser.add_argument("--q_value_memorize_freq", type=int, default=10, help="Frequency for Q-value logging")
    parser.add_argument("--eval_update_freq", type=int, default=512, help="Frequency for evaluation updates")
    
    # Model configuration
    parser.add_argument("--alpha", type=float, default=0.5, help="Memory error weight")
    parser.add_argument("--beta", type=int, default=5, help="KL divergence weight")
    
    # Experiment configuration
    parser.add_argument("--dataset", type=str, default="ETHUSDT", help="Trading pair dataset")
    parser.add_argument("--epoch_number", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for training")
    parser.add_argument("--exp", type=str, default="exp1", help="Experiment name")
    parser.add_argument("--num_step", type=int, default=10, help="Number of steps for multi-step learning")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    agent = DQNAgent(args)
    agent.train()
