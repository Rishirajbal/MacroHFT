import pathlib
import sys
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import joblib
from torch.utils.tensorboard import SummaryWriter
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.model.net import *
from MacroHFT.RL.util.utili import get_ada, get_epsilon, LinearDecaySchedule
from MacroHFT.RL.util.replay_buffer import ReplayBuffer_High
from MacroHFT.RL.util.memory import episodicmemory

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADs"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size", type=int, default=1000000)
parser.add_argument("--dataset", type=str, default="ETHUSDT")
parser.add_argument("--q_value_memorize_freq", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--eval_update_freq", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start", type=float, default=0.7)
parser.add_argument("--epsilon_end", type=float, default=0.3)
parser.add_argument("--decay_length", type=int, default=5)
parser.add_argument("--update_times", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost", type=float, default=0.2 / 1000)
parser.add_argument("--back_time_length", type=int, default=1)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--n_step", type=int, default=1)
parser.add_argument("--epoch_number", type=int, default=15)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=int, default=5)
parser.add_argument("--exp", type=str, default="exp1")
parser.add_argument("--num_step", type=int, default=10)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DQN:
    def __init__(self, args):
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
            
        self.result_path = os.path.join("./result/high_level", args.dataset, args.exp)
        self.model_path = os.path.join(self.result_path, f"seed_{self.seed}")
        
        # Data paths
        self.train_data_path = '/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_train.csv'
        self.val_data_path = '/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_validate.csv'
        self.test_data_path = '/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_test.csv'
        
        self.dataset = args.dataset
        self.num_step = args.num_step
        self.max_holding_number = 0.2 if "ETH" in self.dataset else 0.01
        
        # Training parameters
        self.epoch_number = args.epoch_number
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.n_step = args.n_step
        self.eval_update_freq = args.eval_update_freq
        self.buffer_size = args.buffer_size
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.decay_length = args.decay_length
        self.epsilon_scheduler = LinearDecaySchedule(
            start_epsilon=self.epsilon_start,
            end_epsilon=self.epsilon_end,
            decay_length=self.decay_length
        )
        self.epsilon = args.epsilon_start
        
        # Initialize environment
        self.tech_indicator_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.tech_indicator_list_trend = []
        self.clf_list = []
        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)

        # Initialize networks
        self._initialize_networks(args)
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()
        
        # Initialize memory and buffer
        self.memory = episodicmemory(4320, 5, self.n_state_1, self.n_state_2, 64, self.device)
        self.replay_buffer = ReplayBuffer_High(args, self.n_state_1, self.n_state_2, self.n_action)
        
        # Logging
        self.log_path = os.path.join(self.model_path, "log")
        os.makedirs(self.log_path, exist_ok=True)
        self.writer = SummaryWriter(self.log_path)
        self.update_counter = 0
        self.q_value_memorize_freq = args.q_value_memorize_freq

    def _initialize_networks(self, args):
        """Initialize all neural networks"""
        # Subagents
        self.slope_agents = [
            subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device),
            subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device),
            subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        ]
        self.vol_agents = [
            subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device),
            subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device),
            subagent(self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        ]
        
        # Hyperagent
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())

    def train(self):
        """Main training loop"""
        best_return_rate = -float('inf')
        best_model = None
        
        for epoch in range(1, self.epoch_number + 1):
            print(f'epoch {epoch}')
            
            # Load training data
            df_train = pd.read_csv(self.train_data_path)
            train_env = Training_Env(
                df=df_train,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=random.choice([0, 1]),
                alpha=0
            )
            
            # Training episode
            state, state_trend, state_clf, info = train_env.reset()
            episode_reward = 0
            
            while True:
                action = self.act(state, state_trend, state_clf, info)
                next_state, next_state_trend, next_state_clf, reward, done, next_info = train_env.step(action)
                
                # Store transition
                self._store_transition(
                    state, state_trend, state_clf, info, action, reward,
                    next_state, next_state_trend, next_state_clf, next_info, done
                )
                
                episode_reward += reward
                state, state_trend, state_clf, info = next_state, next_state_trend, next_state_clf, next_info
                
                # Update networks periodically
                if len(self.replay_buffer) > self.batch_size and self.update_counter % self.eval_update_freq == 0:
                    self._update_networks()
                
                if done:
                    break
            
            # Validation and model saving
            self._log_epoch_results(epoch, train_env)
            val_return = self._validate(epoch)
            
            if val_return > best_return_rate:
                best_return_rate = val_return
                best_model = self.hyperagent.state_dict()
                
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch)
        
        # Save best model
        self._save_best_model(best_model)
        self._test_final_model()

    def _store_transition(self, state, state_trend, state_clf, info, action, reward,
                         next_state, next_state_trend, next_state_clf, next_info, done):
        """Store experience in replay buffer"""
        hs = self._calculate_hidden(state, state_trend, info)
        q = reward + self.gamma * (1 - done) * self.q_estimate(next_state, next_state_trend, next_state_clf, next_info)
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
        """Perform network updates"""
        for _ in range(self.update_times):
            td_error, memory_error, kl_loss, q_eval, q_target = self.update(self.replay_buffer)
            
            if self.update_counter % self.q_value_memorize_freq == 1:
                self._log_training_metrics(td_error, memory_error, kl_loss, q_eval, q_target)
                
        if len(self.replay_buffer) > 4320:
            self.memory.re_encode(self.hyperagent)

    # ... (other methods remain the same with proper indentation) ...


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    agent = DQN(args)
    agent.train()
