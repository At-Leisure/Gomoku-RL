import random
import pickle
import string
import math
import pathlib
import dataclasses
import warnings
from typing import Generic, TypeVar, Any, overload, Callable, Optional
#
# import numpy as np
import treelib
import torch
from PIL import Image
from matplotlib import pyplot as plt
#
from _internal.utils.typing import Faction, Node
from _internal.mcts.kernel import GameKernel
from _internal.env.gomoku import GomokuEnv


@dataclasses.dataclass
class Action:
    x: int = -1
    y: int = -1
    f: Faction = Faction.NULL

    # def random(self):
    #     self.x
    #     return self


class NodeData:

    def __init__(self, action: Action | None = None, parent: 'Node[NodeData] | None' = None) -> None:
        """ 当前节点的数据，如果参数都是 None，则表示为根节点"""
        if action == None:
            self.action = Action()
        else:
            self.action = action

        self.parent = parent
        """ 父节点 """

        """ 行为 """
        self.won: int = 0
        """ 赢得次数 """
        self.failed: int = 0
        """ 输的次数 """
        self.total: int = 0
        """ 总的次数 """
        self.weight: float = 0
        """ 权重 """
        self._UCB: float = 0
        """ UCB值 """
        self._valued_after_updated = False
        """ 更新数值后是否更新价值 """

    def set(self, won: int, failed: int, total: int, weight: float):
        self.won = won
        self.failed = failed
        self.total = total
        self.weight = weight
        self._valued_after_updated = False
        return self

    def add(self, *, won: int | None = None, failed: int | None = None,
            total: int | None = None, weight: float | None = None):
        if not won is None:
            self.won += won
        if not failed is None:
            self.failed += failed
        if not total is None:
            self.total += total
        if not weight is None:
            self.weight += weight
        self._valued_after_updated = False
        return self

    def _update_value(self, parent: 'Node[NodeData] | None'):
        if not parent is None:
            self._UCB = self.won / self.total + self.weight * math.sqrt(math.log10(parent.data.total)/self.total)
        else:
            self._UCB = math.inf
        self._valued_after_updated = True
        return self

    @property
    def value(self):
        """ 获取价值。如果没有更新价值，就及时更新价值；否则直接取值 """
        if not self._valued_after_updated:
            self._update_value(self.parent)
        return self._UCB

    @property
    def info(self):
        return self.value, self.won, self.failed, self.total


NT = Node[NodeData]
Tensor = torch.Tensor


class PolicyMakerABC:

    def generate_prob(self, state: Tensor) -> Tensor:
        """ 输入状态输出相同形状的概率 """
        raise NotImplementedError()

    def simulate(self, current_action: Action, state: Tensor) -> tuple[int, int, int]:
        """ 输出模拟结果（不同的决策器自定义模拟过程） 

        - `current_action` 即将下的位置，实际没有落子"""
        raise NotImplementedError()


class RandomPolicyMaker(PolicyMakerABC):

    def __init__(self, c, w, h) -> None:
        self.gk = GameKernel(c, w, h)
        super().__init__()

    def generate_prob(self, state: Tensor) -> Tensor:
        if isinstance(state, Tensor):
            if not state.dtype == torch.uint8:
                raise TypeError()
            else:
                ret = torch.rand(state.shape)
                ret[state != Faction.NULL.value] = 0
                return ret
        else:
            warnings.warn(f'未支持的数组类型{type(state)}')
            raise NotImplementedError()

    def simulate(self, current_action: Action, state: Tensor) -> tuple[int, int, int]:
        """ `state` 是2维数组 shape=[w,h] """
        # state = state.copy_()
        # if not state.ndim == 3:
        #     raise TypeError()

        # ca = current_action
        # state[ca.x, ca.y] = ca.f.value
        # fac = ca.f
        # x, y = ca.x, ca.y

        # while True:
        #     x_vec = torch.tensor(ca.x)
        #     y_vec = torch.tensor(ca.y)
        #     c_vec = torch.arange(self.gk.c)

        #     self.gk.matrix = torch.tensor(state, dtype=self.gk.hyper_matrix.dtype)

        #     if self.gk.detect(c_vec, x_vec, y_vec):
        #         print(self.gk.matrix[c_vec, x_vec, y_vec], self.gk.matrix)
        #         break
        #     else:
        #         if not torch.any(state == 0):
        #             break

        #     prob = self.generate_prob(state)

        #     fac = Faction.next(fac)
        #     x, y = torch.unravel_index(prob.argmax(), prob.shape)
        #     state[x, y] = fac.value

        return random.randint(0, 10), random.randint(0, 10), random.randint(20, 40)


@dataclasses.dataclass
class TreeDate:
    board_wid: int = 6
    board_hei: int = 6
    weight: float = 2


class Tree(treelib.Tree):

    def __init__(self, root: NT, data: TreeDate,
                 tree=None, deep=False, node_class=None, identifier=None):
        super().__init__(tree, deep, node_class, identifier)
        self.data = data
        self.add_node(root)
        self.root_node = root

    def step1_selection(self) -> tuple[Node[NodeData], Tensor]:
        node = self.root_node
        state = torch.zeros([self.data.board_wid, self.data.board_hei], dtype=torch.uint8)
        while True:
            if node.is_leaf(self.identifier):
                return node, state
            else:
                node = max(self.children(node.identifier), key=lambda n: n.data.value)
                state[node.data.action.x, node.data.action.y] = node.data.action.f.value

    def step2_expansion(self, parent_node: NT,
                        current_faction: Faction,
                        state: Tensor,
                        policy_maker: PolicyMakerABC):
        """ 

        - `current_faction` 即将下的派系，实际没有落子 """
        x_vec, y_vec = torch.where(state == Faction.NULL.value)

        if torch.any(x_vec):  # 游戏未结束
            for i in range(x_vec.shape[0]):
                new_node = Node(data=NodeData(Action(int(x_vec[i]),
                                                     int(y_vec[i]),
                                                     current_faction),
                                              parent_node))
                self.add_node(new_node, parent_node)
                w, f, t = self.step3_simulation(new_node.data.action,
                                                state,
                                                policy_maker)
                new_node.data.set(w, f, t, self.data.weight)
                self.step4_backpropagation(new_node)
        else:  # 游戏结束
            ...

    def step3_simulation(self, current_action: Action, state: Tensor,
                         policy_maker: PolicyMakerABC) -> tuple[int, int, int]:
        """ 

        - `current_action` 确定即将下的位置，实际没有落子"""
        return policy_maker.simulate(current_action,
                                     state)

    def step4_backpropagation(self, leaf: NT):
        node = self.parent(leaf.identifier)
        while True:
            if node is None:
                return
            if not node is None:
                if node.data.action.f == leaf.data.action.f:
                    node.data.add(won=leaf.data.won,
                                  failed=leaf.data.failed,
                                  total=leaf.data.total)
                else:
                    node.data.add(won=leaf.data.failed,
                                  failed=leaf.data.won,
                                  total=leaf.data.total)

                node = self.parent(node.identifier)

    def get_node(self, nid) -> NT | None:
        return super().get_node(nid)

    def add_node(self, node: NT, parent=None):
        return super().add_node(node, parent)

    def parent(self, nid) -> NT | None:
        return super().parent(nid)

    def children(self, nid) -> list[NT]:
        return super().children(nid)

    def pickle_dump(self, fn: str | pathlib.Path):
        fn = pathlib.Path(fn)
        with open(fn, 'wb') as f:
            pickle.dump(self, f)
            print(f'树结构已经保存 {fn.relative_to(".")}')

    @classmethod
    def pickle_load(cls, fn: str | pathlib.Path) -> 'Tree':
        fn = pathlib.Path(fn)
        with open(fn, 'rb') as f:
            tree = pickle.load(f)
        if not isinstance(tree, Tree):
            warnings.warn('类型加载错误', RuntimeWarning, 2)
        print(f'树结构成功加载 {fn.relative_to(".")}')
        return tree

    def show_value(self, data_property='value'):
        return self.show(stdout=False, data_property=data_property)

    @property
    def raw_trainingdata(self) -> tuple[Tensor, Tensor]:
        """ 根据树的数据，生成初步的训练数据 """
        data = list[Tensor]()
        target = list[Tensor]()

        init_state = torch.zeros([self.data.board_wid, self.data.board_hei], dtype=torch.uint8)
        env = GomokuEnv((self.data.board_wid, self.data.board_hei), 'rgb_array')

        def make_data(parent_node: NT | None, current_node: NT, current_state: Tensor):
            children = self.children(current_node.identifier)
            if not children:
                return

            # 输入和输出
            dat = torch.zeros((1, 4, self.data.board_wid, self.data.board_hei), dtype=torch.uint8)
            tar = torch.zeros((1, self.data.board_wid, self.data.board_hei))

            m = torch.zeros((self.data.board_wid, self.data.board_hei), dtype=torch.uint8)
            m[current_state == Faction.map(1).value] = 1
            dat[0, 0] = m  # 阵营1

            m = torch.zeros((self.data.board_wid, self.data.board_hei), dtype=torch.uint8)
            m[current_state == Faction.map(2).value] = 1
            dat[0, 1] = m  # 阵营2

            m = torch.zeros((self.data.board_wid, self.data.board_hei), dtype=torch.uint8)
            if not parent_node is None:
                m[parent_node.data.action.x, parent_node.data.action.y] = 1
            dat[0, 2] = m  # 历史落子

            if current_node.data.action.f is Faction.WHITE:
                m = torch.zeros((self.data.board_wid, self.data.board_hei), dtype=torch.uint8)
            else:
                m = torch.ones((self.data.board_wid, self.data.board_hei), dtype=torch.uint8)
            dat[0, 3] = m  # 黑子是否先手

            for current_node in children:
                tar[0, current_node.data.action.x, current_node.data.action.y] = current_node.data.value

            data.append(dat)
            target.append(tar)

            # env.chessboard = current_state.numpy()
            # fn = ''.join(random.choice(string.ascii_letters) for i in range(10))
            # Image.fromarray(env.render()).save(f"./temp/img/traindata/{fn}-state.png")
            # Image.fromarray(env.render()).save(f"{fn}-state.png")

            for child in [c for c in children if 0 != len(self.children(c.identifier))]:
                s = current_state.clone()
                s[child.data.action.x, child.data.action.y] = child.data.action.f.value
                make_data(child, current_node, s)

        make_data(None, self.root_node, init_state)
        # print(*data, sep='\n')
        print('data\'s num =', len(data))
        data_, target_ = torch.cat(data), torch.cat(target)
        print(data_.shape, target_.shape)
        return data_, target_


if __name__ == '__main__':
    root = Node[NodeData](data=NodeData())
    t = Tree(root, TreeDate(3, 3))

    rpm = RandomPolicyMaker(1, 3, 3)

    # state = torch.zeros([3, 3], dtype=torch.uint8)
    fac = Faction.BLACK

    for i in range(1000):
        n, state = t.step1_selection()
        t.step2_expansion(n,
                          fac,
                          state,
                          rpm)
        fac = Faction.next(fac)

    # print(t.show_value('info'))
    # t.pickle_dump('./temp/2.tree')
    d, t = t.raw_trainingdata

    print(d[d.shape[0]//2], t[d.shape[0]//2])
