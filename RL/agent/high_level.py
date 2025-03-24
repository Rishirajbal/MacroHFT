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
warnings.filterwarnings("ignore")

# Set repository root path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

# Import from repository modules
from MacroHFT.model.net import subagent, hyperagent
from MacroHFT.env.high_level_env import Testing_Env, Training_Env
from MacroHFT.RL.util.replay_buffer import ReplayBuffer_High
from MacroHFT.RL.util.memory import episodicmemory

# Threading configuration
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

class LinearDecaySchedule:
    def __init__(self, start_epsilon, end_epsilon, decay_length):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_length = decay_length

    def get_epsilon(self, epoch):
        return max(self.end_epsilon, 
                 self.start_epsilon - (self.start_epsilon-self.end_epsilon)*epoch/self.decay_length)

class DQNAgent:
    def __init__(self, args):
        self.args = args
        self._setup_device()
        self._setup_paths()
        self._load_technical_indicators()
        self._init_parameters()
        self._init_networks()
        self._setup_training()
        
    def _setup_device(self):
        self.device = torch.device(self.args.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        
    def _setup_paths(self):
        self.result_path = REPO_ROOT/'result/high_level'/self.args.dataset/self.args.exp
        self.model_path = self.result_path/f"seed_{self.args.seed}"
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Data paths - modify these according to your data location
        self.train_data_path = REPO_ROOT/'data'/self.args.dataset/'whole/df_train.csv'
        self.val_data_path = REPO_ROOT/'data'/self.args.dataset/'whole/df_validate.csv'
        self.test_data_path = REPO_ROOT/'data'/self.args.dataset/'whole/df_test.csv'
        
    def _load_technical_indicators(self):
        try:
            self.tech_indicator_list = np.load(REPO_ROOT/'data/feature_list/single_features.npy', 
                                             allow_pickle=True).tolist()
            self.tech_indicator_list_trend = np.load(REPO_ROOT/'data/feature_list/trend_features.npy', 
                                                   allow_pickle=True).tolist()
        except FileNotFoundError:
            self.tech_indicator_list = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.tech_indicator_list_trend = ['Adj Close']
            
        self.clf_list = ['slope_360', 'vol_360']
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        
    def _init_networks(self):
        # Subagents
        self.slope_agents = nn.ModuleList([
            subagent(self.n_state_1, self.n_state_2, 2, 64).to(self.device) 
            for _ in range(3)
        ])
        self.vol_agents = nn.ModuleList([
            subagent(self.n_state_1, self.n_state_2, 2, 64).to(self.device)
            for _ in range(3)
        ])
        
        # Load pretrained models from repo
        for i in range(3):
            slope_path = REPO_ROOT/f"result/low_level/{self.args.dataset}/best_model/slope/{i+1}/best_model.pkl"
            vol_path = REPO_ROOT/f"result/low_level/{self.args.dataset}/best_model/vol/{i+1}/best_model.pkl"
            
            if slope_path.exists():
                self.slope_agents[i].load_state_dict(torch.load(slope_path, map_location=self.device))
            if vol_path.exists():
                self.vol_agents[i].load_state_dict(torch.load(vol_path, map_location=self.device))
        
        # Hyperagent
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, 2, 32).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, 2, 32).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
        
    def _setup_training(self):
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(), lr=self.args.lr)
        self.loss_func = nn.MSELoss()
        self.epsilon_scheduler = LinearDecaySchedule(
            self.args.epsilon_start, self.args.epsilon_end, self.args.decay_length)
        self.epsilon = self.args.epsilon_start
        self.memory = episodicmemory(4320, 5, self.n_state_1, self.n_state_2, 64, self.device)
        self.replay_buffer = ReplayBuffer_High(self.args, self.n_state_1, self.n_state_2, 2)
        self.writer = SummaryWriter(self.model_path/"logs")
        
    def train(self):
        best_return = -float('inf')
        
        for epoch in range(1, self.args.epoch_number+1):
            df_train = pd.read_csv(self.train_data_path)
            env = Training_Env(df_train, 
                             tech_indicator_list=self.tech_indicator_list,
                             max_holding_number=self._get_max_holding(),
                             transcation_cost=self.args.transcation_cost)
            
            state = env.reset()
            done = False
            
            while not done:
                action = self._select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Store transition and update
                self._store_transition(state, action, reward, next_state, done, info)
                if len(self.replay_buffer) > self.args.batch_size:
                    self._update_networks()
                
                state = next_state
            
            # Validation and model saving
            val_return = self._validate()
            if val_return > best_return:
                best_return = val_return
                torch.save(self.hyperagent.state_dict(), self.model_path/"best_model.pkl")
                
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch)
        
        self._test()

    # ... (include all other methods from your original file)
    # Make sure to update paths to use REPO_ROOT where needed

def parse_args():
    parser = argparse.ArgumentParser()
    # ... (keep your existing argument parser setup)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    agent = DQNAgent(args)
    agent.train()
