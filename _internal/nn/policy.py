""" 策略 神经网络 """
import os
import copy
import random
import time
from typing import Literal
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

from ..utils.abc import PlayerABC


class ValueNetwork_other(nn.Module):
    """policy-value network module"""

    def __init__(self, board_size: tuple[int, int]):
        super().__init__()
        self.board_width, self.board_height = board_size

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


class ValueNetworkV0(nn.Module):
    """复制别人的网络结构"""

    def __init__(its, board_size: tuple[int, int]):
        super().__init__()
        its.board_width, its.board_height = board_size

        its.common_block = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        its.action_block = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * its.board_width * its.board_height,
                      its.board_width * its.board_height),
            nn.LogSoftmax(),
        )
        its.svalue_block = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * its.board_width * its.board_height, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        """ state-value layers """

    def forward(its, state_input):
        common_output = its.common_block(state_input)
        action_output = its.action_block(common_output)
        svalue_output = its.svalue_block(common_output)

        return action_output, svalue_output


class ValueNetworkV1(ValueNetworkV0):

    def __init__(its, board_size: tuple[int, int]):
        raise NotImplementedError()
        super().__init__()
        its.board_width, its.board_height = board_size
        its.common_block = nn.Sequential(
        )
        its.action_block = nn.Sequential(
        )
        its.svalue_block = nn.Sequential(
        )


class Policymaker:
    """ 决策器 """

    def __init__(self,
                 board_size: tuple[int, int],
                 device: Literal['cpu', 'cuda'],
                 from_: str = None,) -> None:
        self.device = torch.device(device)
        self.net = ValueNetworkV0(board_size)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    weight_decay=1e-4)
        if not from_ is None:
            self.load_model(from_)

    def load_model(self, source: str):
        if not source.endswith('.gmk.pt'):
            print(f'Warning: postfix should end with `.gmk.pt`, but {source}')
        params = torch.load(source)
        self.net.load_state_dict(params)

    def save_model(self, target: str):
        """ 保存模型
        >>> self.save_model('./models/test.gmk.pt)"""
        if not target.endswith('.gmk.pt'):
            print(f'Warning: postfix should end with `.gmk.pt`,but {target}')
        torch.save(self.net.state_dict(), target)

    def train_once(self, state_batch, mcts_probs, winner_batch, lr: float):
        """ perform a training step """
        # Pytorch已经可以自动回收我们不用的显存，类似于python的引用机制，当某一内存内的数据不再有任何变量引用时，这部分的内存便会被释放。
        state_batch = Variable(torch.FloatTensor(state_batch).to(self.device))
        mcts_probs = Variable(torch.FloatTensor(mcts_probs).to(self.device))
        winner_batch = Variable(torch.FloatTensor(winner_batch).to(self.device))

        self.optimizer.zero_grad()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # forward 网络预测
        logged_action_probs, value = self.net(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * logged_action_probs, 1))
        loss = value_loss + policy_loss

        # backward 更新参数
        loss.backward()
        # z.backward() 会计算 z 相对于 x 的梯度，并将这些梯度存储在 x.grad 中。
        self.optimizer.step()
        # optimizer.step() 会根据 x.grad 优化更新 x 的值。

        # 计算策略熵，仅用于监控
        entropy = -torch.mean(torch.sum(torch.exp(logged_action_probs) * logged_action_probs, 1))
        # 返回损失值和熵值
        return loss.item(), entropy.item()


class NNPlayer(PlayerABC):
    ...
