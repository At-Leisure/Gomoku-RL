import abc
from typing import Any, Tuple

import gym


class ActionABC(abc.ABC):
    ...


class PlayerABC(abc.ABC):
    """ PlayerABC (Player for Abstract Basic Class)  """

    @abc.abstractmethod
    def get_action(self,):
        raise NotImplementedError()


class EnvABC(gym.Env, abc.ABC):
    """ EnvironmentABC (Environment for Abstract Basic Class)  """

    def __init__(self) -> None:
        super().__init__()
        self._current_agent: Any = None

    @property
    def current_agent(self):
        """ 当前`执棋者` ，在 step() 后会被调用"""
        if self._current_agent is None:
            raise TypeError()
        else:
            return self._current_agent

    @current_agent.setter
    def current_agent(self, agnet):
        self._current_agent = agnet

    @abc.abstractmethod
    def step(self, action: Any, next_agent=None) -> Tuple[Any, float, bool, bool, dict]: ...

    @abc.abstractmethod
    def is_won(self,):
        """ 返回游戏状态是否已经获胜，返回 [True, 获胜方 | None] 或是 [False, None] 
        >>> is_end,winner = self.is_end()"""


class MCTSABC(abc.ABC):
    """ MCTS 的抽象基类 """
    @abc.abstractmethod
    def FN_policy_value(self):
        """ 策略价值函数，用于给出每个位置的策略价值。一个接受状态并输出(动作，概率)元组列表和状态分数的函数 """
    @abc.abstractmethod
    def FN_rollout_policy(self):
        """ 推出价值函数(随机性)。在推出阶段使用的policy_fn的粗略、快速版本。 """
    @abc.abstractmethod
    def S1_select(self):
        """ step1 选择:  在子节点中选择动作值Q最大的动作加上u(P)。返回:(action, next_node)的元组

        `node` 进行选择的节点，None 表示从根节点进行选择，返回的是子节点

        如果要选择叶子节点，需要在外部引用
        >>> node = mcts.root
            while not node.is_leaf(mcts._tree.identifier):
                node = mcts.S1_select(node)"""
    @abc.abstractmethod
    def S2_expand(self):
        """ step2 扩展: 从根到叶运行一次扩散，获取值这片叶子通过它的亲本繁殖回来。状态是就地修改的，因此必须提供副本。 """
    @abc.abstractmethod
    def S3_simulate(self):
        """ step3 模拟:  """
    @abc.abstractmethod
    def S4_backward(self):
        """ step4 反向传播:  """
