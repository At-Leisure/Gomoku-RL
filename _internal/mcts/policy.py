""" 策略 MCTS (Monte Carlo Tree Search) 蒙特卡洛树搜索 """

from typing import overload, Generic, TypeVar, Any, Callable
from functools import cached_property, partial
import numpy as np
import dataclasses
import math
from copy import deepcopy

import gym

from ..utils.abc import MCTSABC, PlayerABC, EnvABC
from ..utils.typing import Tree, Node
from .kernel import GameKernel


@dataclasses.dataclass
class Action:
    x: int = None
    y: int = None
    faction: int = None


@dataclasses.dataclass
class DataBase:
    action: Action
    win_times: int = 0  # 模拟获胜次数
    sim_times: int = 0  # 当前结点的总模拟次数
    parent_times: int = 0  # 父结点的总模拟次数
    _u: int = 0
    _P: bool = None  # 先验概率
    is_backwarded: bool = False  # 一个节点只后向传播一次


class NodeData(DataBase):

    def __init__(self, action: Action, prior_prob: float = None) -> None:
        super().__init__(action=action, _P=prior_prob)

    def value(self, c: float):
        """计算并返回该节点的值。
        它是叶子值Q和这个节点的先验的组合
        根据访问量进行调整后，美国。
        C: (0, inf)中的一个数字，控制的相对影响
        值Q和先验概率P。
        """
        if self.sim_times != 0:
            a = self.win_times / self.sim_times
            b = c * self._P * np.sqrt(np.log(self.parent_times)/self.sim_times)
            return a+b
        else:
            return math.inf


NodeType = Node[NodeData]
TreeType = Tree[NodeType]


class MCTSTree(Tree[NodeType]):

    def select(self, from_: NodeType, c: int):
        """ """
        node = max(self.children(from_.identifier),
                   key=lambda n: n.data.value)
        return node.data.action, node


class MCTSBase(MCTSABC):

    @property
    def root(self):
        return self._tree.get_node(self._tree.root)

    def __init__(self, c_weight: float = 2.0) -> None:
        """ 
        `policy_value_fn` 策略价值函数
        `c_puct`（0， inf） 中的一个数字，用于控制探索收敛到最大值策略的速度。更高的值意味着依赖先前的更多"""
        self._tree = TreeType()
        root = NodeType(data=NodeData(Action(), 1.0))
        self._tree.add_node(root)  # add root node
        self._c = c_weight

    def FN_policy_value(self, state: np.ndarray):
        """ 策略价值函数(平均性)，用于给出每个位置的策略价值。一个接受状态并输出(动作，概率)元组列表和状态分数的函数 

        为了行动的有效性，使在 state 不为 0 的位置的概率为 0，其他位置的概率均匀分配，所有概率之和为 1"""
        action_probs = np.ones(state.shape) / np.sum(state != 0)
        action_probs[state != 0] = 0
        return action_probs

    def FN_rollout_policy(self, state: np.ndarray):
        """ 推出策略函数(随机性)。在推出阶段使用的policy_fn的粗略、快速版本。 

        为了行动的有效性，使在 state 不为 0 的位置的概率为 0，其他位置的概率随机分配"""
        action_probs = np.random.random(state.shape)
        action_probs[state != 0] = 0
        return action_probs

    def S1_select(self, node: NodeType = None):
        if node is None:
            node = self.root
        if node.is_leaf(self._tree.identifier):
            print('不能对叶子节点进行`选择`')
            return node

        def key(n: NodeType):
            return n.data.value(self._c)
        child = max(self._tree.children(node.identifier), key=key)
        return child  # 返回子节点

    def S2_expand(self, node: NodeType, state: np.ndarray):
        # 扩展叶子节点，默认
        children_action = [n.data.action for n in self._tree.children(node.identifier)]
        if state.ndim == 2:
            a, b = np.where(state == 0)
            for x, y in zip(a, b):
                act = Action(x, y, None)  # TODO 更改faction
                if not act in children_action:
                    self._tree.add_node(NodeType(data=NodeData(act, 1.0)), node)
        else:
            raise NotImplementedError()

    def S3_simulate(self, node: NodeType, env: EnvABC, max_iter=1000):
        """ step3 模拟:  使用推出策略玩到游戏结束 TODO 验证S3的有效性

        如果当前玩家获胜则返回+1，如果对手获胜则返回-1，如果超过最大迭代次数视为平局，平局时返回0"""
        env = env.copy()  # 需要使用副本进行模拟
        player = env.current_agent
        for i in range(max_iter):
            is_end, winner = env.is_won()
            if is_end:
                break
            action_probs = self.FN_rollout_policy(env.chessboard)
            # 将一维索引转换为二维索引
            x, y = np.unravel_index(action_probs.argmax(), action_probs.shape)
            env.step(Action(x, y, 1 if env.current_agent == 1 else 2),
                     next_agent=2 if env.current_agent == 1 else 1)  # TODO 更改faction,agent
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")

        node.data.sim_times += 1
        if winner == player:
            node.data.win_times += 1
        if winner is None:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def S4_backward(self, node: NodeType):
        if node.data.is_backwarded:
            print(f'Warning: node {node} is backwarded')
        else:
            node.data.is_backwarded = True

        while not node.is_root(self._tree.identifier):
            super_node = self._tree.parent(node.identifier)
            super_node.data.win_times += node.data.win_times
            super_node.data.sim_times += node.data.sim_times
            node = super_node


class MCTSPlayer(PlayerABC):
    pass
