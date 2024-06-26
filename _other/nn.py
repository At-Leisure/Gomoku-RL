from torchinfo import summary
import os
import copy
import random
import time
from operator import itemgetter
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import gym
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt
from IPython import display


class NetBase(nn.Module):

    def __init__(self, wid: int, hei: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.board_width, self.board_height = wid, hei


class NormalCNN(NetBase):
    """policy-value network module"""

    def __init__(self, wid: int, hei: int):
        super().__init__(wid, hei)
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * self.board_width * self.board_height,
                                 self.board_width * self.board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * self.board_width * self.board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet:
    """policy-value network """

    def __init__(self, Module: type[NetBase], model_file: str = None):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module

        self.policy_value_net = Module().to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.policy_value_net.board_width, self.policy_value_net.board_height))
        log_act_probs, value = self.policy_value_net(
            Variable(torch.from_numpy(current_state)).to(self.device).float())
        act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
        mcts_probs = Variable(torch.FloatTensor(mcts_probs).to(self.device))
        winner_batch = Variable(torch.FloatTensor(winner_batch).to(self.device))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )
        # return loss.data, entropy.data
        # for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)


class NormalVGG(NetBase):
    """policy-value network module"""

    def __init__(self, wid, hei):
        super().__init__(wid, hei)

        self.common = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),    # (224 - 3 + 2*1) / 1 + 1 = 224
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),   # (224 - 3 + 2*2) / 1 + 1 = 224
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),    # (224 - 3 + 2*1) / 1 + 1 = 224
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),   # (224 - 3 + 2*2) / 1 + 1 = 224
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * self.board_width * self.board_height, self.board_width * self.board_height),
            nn.LogSoftmax(),
        )
        self.value = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.board_width * self.board_height, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        c = self.common(x)
        p = self.policy(c)
        v = self.value(c)
        return p, v


class NormalResNet(NetBase):
    """policy-value network module"""

    class BasicBlock(nn.Module):

        def __init__(self, main_channel: nn.Module, shortcut: nn.Module = None) -> None:
            super().__init__()
            self.main_channel = main_channel
            self.shortcut = shortcut

        def forward(self, x):
            iden = x if self.shortcut is None else self.shortcut(x)
            out = self.main_channel(x)
            return out + iden

    def __init__(self, wid, hei):
        super().__init__(wid, hei)

        self.common = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),
            self.BasicBlock(
                nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            self.BasicBlock(
                nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * self.board_width * self.board_height, self.board_width * self.board_height),
            nn.LogSoftmax(),
        )
        self.value = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.board_width * self.board_height, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        c = self.common(x)
        p = self.policy(c)
        v = self.value(c)
        return p, v


class NormalDenseNet(NetBase):
    """policy-value network module"""
    class Block(nn.Module):
        def __init__(self, n_layers, n_channels) -> None:
            super().__init__()
            self.n_layers = n_layers
            self.ms = nn.ModuleList([
                nn.Conv2d(
                    n_channels*i,
                    n_channels,
                    kernel_size=3, stride=1, padding=1)
                for i in range(1, 1+n_layers)
            ])

        def forward(self, x):
            input = x
            output = None
            for m in self.ms:
                output = m(input)
                input = torch.cat([output, input], dim=1)  # 按通道合并
            return output

    def __init__(self, wid, hei):
        super().__init__(wid, hei)

        self.common = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1),
            self.Block(3, 32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            self.Block(3, 64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        )
        self.policy = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * self.board_width * self.board_height, self.board_width * self.board_height),
            nn.LogSoftmax(),
        )
        self.value = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.board_width * self.board_height, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # c = self.common(x);return c

        c = self.common(x)
        p = self.policy(c)
        v = self.value(c)
        return p, v
