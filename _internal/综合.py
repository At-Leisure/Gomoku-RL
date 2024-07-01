import math

import torch

from .env.gomoku import GomokuEnv
from .env.action import Action
from .mcts.policy import MCTSBase, NodeData
from .mcts.kernel import GameKernel
from .utils.typing import Node, Faction

# import _torch

# a = _torch.tensor([1,2],_torch.float64,_torch.cpu)

class MCTSGomoku(MCTSBase):
    """ 专用于 Gomoku 的 MCTS """

    def __init__(self, board_wid: int,
                 board_hei: int,
                 n_simulate: int = 20,
                 c_weight: float = 2) -> None:
        super().__init__(c_weight)
        self.n_simulate = n_simulate
        self.gk = GameKernel(n_simulate, board_wid, board_hei)
        self.env = GomokuEnv((board_wid, board_hei))
        self.players = [1, 2]

    def FN_rollout_policy(self, state: torch.Tensor):
        action_probs = torch.rand(state.shape)
        action_probs[state != 0] = 0
        return action_probs

    def S3_simulate(self, node: Node[NodeData], env: GomokuEnv, max_iter=100):
        # return super().S3_simulate(node, env, max_iter)
        self.gk.matrix = torch.tensor(env.chessboard)
        last_player = env.last_action.faction
        current_player = initial_player = self.players[(self.players.index(last_player)+1) % 2]  # 1->2, 2->1
        x_pos = torch.tensor([env.last_action.x]*self.n_simulate, dtype=torch.int
                             ).reshape((self.n_simulate, 1, 1))
        y_pos = torch.tensor([env.last_action.y]*self.n_simulate, dtype=torch.int
                             ).reshape((self.n_simulate, 1, 1))

        players = [1, 2]
        c_pos = torch.arange(x_pos.shape[0]).reshape([-1, 1, 1])
        current_player = 1

        for i in range(100):
            won_mat = self.gk.detect(c_pos, x_pos, y_pos)
            gaming_mat = self.gk.matrix[~won_mat]
            if gaming_mat.shape[0] == 0:
                break
            action_probs = self.FN_rollout_policy(gaming_mat)
            act_arg = action_probs.reshape([action_probs.shape[0], -1]).argmax(dim=1)
            x, y = torch.unravel_index(act_arg, action_probs.shape[1:])
            x, y = x.reshape([action_probs.shape[0], 1, 1]), y.reshape([action_probs.shape[0], 1, 1])
            c = c_pos[~won_mat]
            channel = c.reshape([-1])
            x_pos[channel] = x
            y_pos[channel] = y

            current_player = players[(players.index(current_player)+1) % 2]

            self.gk.matrix[c, x, y] = current_player

    def batch_action_probvec(self, batch_state: torch.Tensor,
                             action_vec: list[Action],
                             max_iter: int = None) -> torch.Tensor:
        """ 返回与当前行为相同批次(1维)的获胜概率向量

        ## Parameters
        - `batch_state` 棋盘状态[uint]，shape[b,,w,h]。
            0位置无棋子，1位置属于派系1，2位置属于派系2
        - `action_vec` 当前行为的向量，shape[b]
        - `max_iter` 最大走子次数，None表示走到尽头

        ## Return
        - `prob_vec` 概率向量，shape[b]"""
        raise NotImplementedError()

    def diffuse_leaf(self, state: torch.Tensor, current_faction: Faction | str | int,
                     last_action: Action = None,  max_iter: int = None) -> torch.Tensor:
        """ 获取当前行为的获胜概率(随机走子或是智能预测)

        扩散叶子节点            TODO 搞一个 [b,c,w,h]
        输入当前棋盘状态，上一个棋子，当前棋子，通过批次计算，依据大数定理，返回一个和棋盘尺寸相同的概率矩阵

        ## Parameters
        - `state` 棋盘状态[int]，0位置无棋子，1位置属于派系1，2位置属于派系2
        - `last_action` 上个棋子的位置和派系，None表示刚开局
        - `current_` 当前棋子的派系，last_action=None时，表示先手棋子的派系
        - `max_iter` 最大走子次数，None表示走到尽头

        ## Return
        - `prob_matrix` 一个和棋盘尺寸相同的矩阵，包含了当前棋子下在各个点的概率，
            其中先前棋子(不可下棋位置)的概率为0，可下位置的概率在[0,1]之间"""
        self.gk.matrix = state
        initial_f = cf = Faction.map(current_faction)  # 当前玩家

        c, w, h = state.shape
        # 通道向量
        c_vec = torch.arange(c, dtype=torch.int).reshape([c, 1, 1])  # shape: [c,1,1]
        if last_action is None:  # 刚开局，先下一子
            x_vec = torch.randint(0, w, [c, 1, 1])  # torch.randint 左闭右开
            y_vec = torch.randint(0, h, [c, 1, 1])
            self.gk.matrix[c_vec, x_vec, y_vec] = cf.value
            cf = Faction.next(cf)
            print('先手阵营:', cf)

        if max_iter is None:
            max_iter = torch.sum(state == 0)

        for i in range(max_iter):
            won_mat = self.gk.detect(c_vec, x_vec, y_vec)
            gaming_mat = self.gk.matrix[~won_mat]
            if gaming_mat.shape[0] == 0:
                break
            action_probs = self.FN_rollout_policy(gaming_mat)
            act_arg = action_probs.reshape([action_probs.shape[0], -1]).argmax(dim=1)
            x, y = torch.unravel_index(act_arg, action_probs.shape[1:])
            x, y = x.reshape([action_probs.shape[0], 1, 1]), y.reshape([action_probs.shape[0], 1, 1])
            c = c_vec[~won_mat]
            channel = c.reshape([-1])
            x_vec[channel] = x
            y_vec[channel] = y

            self.gk.matrix[c, x, y] = cf.value
            cf = Faction.next(cf)
        print(won_mat)
        print(self.gk.ma)
        return c_vec, x_vec, y_vec
