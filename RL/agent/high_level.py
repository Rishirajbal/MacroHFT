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
from MacroHFT.env.high_level_env import Testing_Env, Training_Env
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


class DQN(object):
    def __init__(self, args):
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
            
        self.result_path = os.path.join("./result/high_level", '{}'.format(args.dataset), args.exp)
        self.model_path = os.path.join(self.result_path, "seed_{}".format(self.seed))
        
        # Updated paths
        self.train_data_path = '/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_train.csv'
        self.val_data_path = '/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_validate.csv'
        self.test_data_path = '/content/drive/MyDrive/MacroHFT/data/ETHUSDT/whole/df_test.csv'
        
        self.dataset = args.dataset
        self.num_step = args.num_step
        
        if "BTC" in self.dataset:
            self.max_holding_number = 0.01
        elif "ETH" in self.dataset:
            self.max_holding_number = 0.2
        elif "DOT" in self.dataset:
            self.max_holding_number = 10
        elif "LTC" in self.dataset:
            self.max_holding_number = 10
        else:
            raise Exception("We do not support other datasets yet")
            
        self.epoch_number = args.epoch_number
        self.log_path = os.path.join(self.model_path, "log")
        
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
            
        self.writer = SummaryWriter(self.log_path)
        self.update_counter = 0
        self.q_value_memorize_freq = args.q_value_memorize_freq

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Updated technical indicators to match your dataset
        self.tech_indicator_list = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.tech_indicator_list_trend = []  # No trend features
        self.clf_list = []  # No classification features

        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)  # Should be 6
        self.n_state_2 = len(self.tech_indicator_list_trend)  # Should be 0

        # Initialize subagents
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

        # Initialize hyperagent with proper dimensions
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())

        # Initialize optimizer and loss function
        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()

        # Training parameters
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

        # Initialize memory
        self.memory = episodicmemory(4320, 5, self.n_state_1, self.n_state_2, 64, self.device)

    def calculate_q(self, w, qs):
        q_tensor = torch.stack(qs)
        q_tensor = q_tensor.permute(1, 0, 2)
        weights_reshaped = w.view(-1, 1, 6)
        combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)
        return combined_q

    def update(self, replay_buffer):
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Handle empty state_trend case
        if self.n_state_2 == 0:
            batch['state_trend'] = torch.zeros_like(batch['state'][:, :0])  # Empty tensor with batch dimension
            batch['next_state_trend'] = torch.zeros_like(batch['next_state'][:, :0])

        w_current = self.hyperagent(batch['state'], batch['state_trend'], batch['state_clf'], batch['previous_action'])
        w_next = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])
        w_next_ = self.hyperagent(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])

        qs_current = [
            agent(batch['state'], batch['state_trend'], batch['previous_action']) 
            for agent in self.slope_agents + self.vol_agents
        ]
        
        qs_next = [
            agent(batch['next_state'], batch['next_state_trend'], batch['next_previous_action'])
            for agent in self.slope_agents + self.vol_agents
        ]

        q_distribution = self.calculate_q(w_current, qs_current)
        q_current = q_distribution.gather(-1, batch['action']).squeeze(-1)
        
        a_argmax = self.calculate_q(w_next_, qs_next).argmax(dim=-1, keepdim=True)
        q_nexts = self.calculate_q(w_next, qs_next)
        q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * q_nexts.gather(-1, a_argmax).squeeze(-1)

        td_error = self.loss_func(q_current, q_target)
        memory_error = self.loss_func(q_current, batch['q_memory'])

        demonstration = batch['demo_action']
        KL_loss = F.kl_div(
            (q_distribution.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8),
            reduction="batchmean",
        )

        loss = td_error + args.alpha * memory_error + args.beta * KL_loss
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.hyperagent.parameters(), 1)
        self.optimizer.step()
        
        for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        self.update_counter += 1
        return td_error.cpu(), memory_error.cpu(), KL_loss.cpu(), torch.mean(q_current.cpu()), torch.mean(q_target.cpu())

    def act(self, state, state_trend, state_clf, info):
        # Ensure proper tensor shapes with batch dimension
        x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device) if len(state_trend) > 0 else torch.zeros(1, 0).to(self.device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device) if len(state_clf) > 0 else torch.zeros(1, 0).to(self.device)
        previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0)
        
        if np.random.uniform() < (1 - self.epsilon):
            qs = [
                agent(x1, x2, previous_action)
                for agent in self.slope_agents + self.vol_agents
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action = random.choice([0, 1])
        return action

    def act_test(self, state, state_trend, state_clf, info):
        with torch.no_grad():
            x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device) if len(state_trend) > 0 else torch.zeros(1, 0).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device) if len(state_clf) > 0 else torch.zeros(1, 0).to(self.device)
            previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0)
            
            qs = [
                agent(x1, x2, previous_action)
                for agent in self.slope_agents + self.vol_agents
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            return action[0]

    def q_estimate(self, state, state_trend, state_clf, info):
        with torch.no_grad():
            x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device) if len(state_trend) > 0 else torch.zeros(1, 0).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device) if len(state_clf) > 0 else torch.zeros(1, 0).to(self.device)
            previous_action = torch.unsqueeze(torch.tensor(info["previous_action"]).long().to(self.device), 0)
            
            qs = [
                agent(x1, x2, previous_action)
                for agent in self.slope_agents + self.vol_agents
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            actions_value = self.calculate_q(w, qs)
            q = torch.max(actions_value, 1)[0].item()  # Return scalar value
            return q

    # ... (rest of the class methods remain the same) ...


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    agent = DQN(args)
    agent.train()
