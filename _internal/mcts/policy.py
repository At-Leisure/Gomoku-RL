""" 策略 MCTS (Monte Carlo Tree Search) 蒙特卡洛树搜索 """

from typing import overload, Generic, TypeVar, Any, Callable
from functools import cached_property
import numpy as np

from . import env as _env
from . import policy as _policy
from . import utils as _utils


class NodeData:

    def __init__(self, prior_prob) -> None:
        self.win_times = 0  # 模拟获胜次数
        self.sim_times = 0  # 当前结点的总模拟次数
        self.parent_times = 0  # 父结点的总模拟次数
        self.action = _utils.Action()
        self._u = 0
        self._P = prior_prob

    @cached_property
    def value(self, c: float):
        """计算并返回该节点的值。
        它是叶子值Q和这个节点的先验的组合
        根据访问量进行调整后，美国。
        C: (0, inf)中的一个数字，控制的相对影响
        值Q和先验概率P。
        """
        a = self.win_times / self.sim_times
        b = c * self._P * np.sqrt(np.log(self.parent_times)/self.sim_times)
        return a+b


NodeType = _utils.Node[NodeData]


class MCTSTree(_utils.Tree[NodeType]):

    def select(self, from_: NodeType, c: int):
        """ 在子节点中选择动作值Q最大的动作加上u(P)
        返回:(action, next_node)的元组"""
        node = max(self.children(from_.identifier),
                   key=lambda n: n.data.value)
        return node.data.action, node


class MCTS:

    def __init__(self,
                 policy_value_fn: Callable[[np.ndarray], tuple[np.ndarray, float]],
                 c_puct: int = 5) -> None:
        """ 
        `policy_value_fn` 策略价值函数
        `c_puct`（0， inf） 中的一个数字，用于控制探索收敛到最大值策略的速度。更高的值意味着依赖先前的更多"""
        self._tree = MCTSTree(node_class=NodeType)
        self._tree.add_node(NodeType(data=NodeData(1.0)))  # add root node
        self.policy_value_fn = policy_value_fn
        self._c_puct = c_puct

    def diffuse(self, state: _env.GomokuEnv):
        """ 从根到叶运行一次扩散，获取值这片叶子通过它的亲本繁殖回来。状态是就地修改的，因此必须提供副本。 """
        node = self._tree.root
        tree = self._tree
        while not node.is_leaf(self._tree.identifier):
            # 贪婪地选择下一步
            action, node = tree.select(node, self._c_puct)
            state.step(action)

        action_probs, leaf_value = self.policy_value_fn(state)


class MCTSPlayer(_utils.PlayerABC):

    def get_action(self, env, return_prob=0):
        sensible_moves = env.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board_width * board_width)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temperature)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=noise_eps * probs + (1 - noise_eps) * np.random.dirichlet(
                        dirichlet_alpha * np.ones(len(probs))))
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
