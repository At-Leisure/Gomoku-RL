import typing
import enum
from typing import Generic, TypeVar, Any, overload, Callable, Optional
from dataclasses import dataclass
from functools import wraps

from typing_extensions import Self
import treelib
import torch

_VT = TypeVar("_VT")
""" generic variable type """
_NT = TypeVar("_NT")
""" generic node type """
_DT = TypeVar("_DT")
""" generic data type """


class Node(treelib.Node, Generic[_VT]):
    """ 泛型节点，便于变量类型检查 
    self.data 是泛型变量"""

    def __init__(self, tag=None, identifier=None, expanded=True, data: _VT = None):
        super().__init__(tag, identifier, expanded, data)
        self.data: _VT


class Tree(treelib.Tree, Generic[_NT]):
    """ 泛型树类，便于变量类型检查和错误排查 """

    def __init__(self, tree=None, deep=False, node_class=None, identifier=None):
        super().__init__(tree, deep, node_class, identifier)

    def get_node(self, nid) -> _NT:
        return super().get_node(nid)

    def add_node(self, node: _NT, parent=None):
        return super().add_node(node, parent)

    def parent(self, nid) -> _NT | None:
        return super().parent(nid)

    def children(self, nid) -> list[_NT]:
        return super().children(nid)

    def show(self, nid=None, level=..., idhidden=True, filter=None,
             key: Callable[[_NT], Any] = None,
             reverse=False, line_type="ascii-ex", data_property=None, stdout=True, sorting=True):
        return super().show(nid, level, idhidden, filter,
                            key, reverse, line_type, data_property,
                            stdout, sorting)


# class Tensor(torch.Tensor, Generic[DT]):

#     ...


# def tensor(data: Any, dtype: Optional[DT] = None,
#            device: str | torch.device | int = None,
#            requires_grad: bool = False,
#            pin_memory: bool = False) -> Tensor[DT]:
#     return torch.tensor(data, dtype, device, requires_grad, pin_memory)


class Faction(enum.Enum):
    """ 枚举派系 

    Enum类中的每个成员都有一个唯一的整数值，这个值默认从1开始，并且每次递增1。

    >>> print(Faction.keys(), Faction.values())
    ('NULL', 'WHITE', 'BLACK') (0, 1, 2)"""

    NULL = 0
    WHITE = 1
    BLACK = 2

    @classmethod
    def keys(cls):
        return tuple(cls.__members__.keys())

    @classmethod
    def values(cls):
        return tuple(v.value for v in cls.__members__.values())

    @classmethod
    def _get_index(cls, now: Self | int | str):
        values = cls.values()
        keys = cls.keys()

        if isinstance(now, Faction):
            idx = keys.index(now.name)

        elif isinstance(now, str):
            idx = keys.index(now)

        elif isinstance(now, int):
            idx = values.index(now)
        else:
            raise TypeError()
        return idx

    @classmethod
    def map(cls, now: Self | int | str):
        """ 根据索引映射枚举对象 

        >>> Faction.map(0),Faction.map(1),Faction.map(2)
        Faction.NULL Faction.WHITE Faction.BLACK"""
        keys = cls.keys()
        idx = cls._get_index(now)
        return cls[keys[idx]]

    @classmethod
    def next(cls, now: Self | int | str):
        """ 按值顺序给出下一个枚举对象。1->2->1 or 1->2->3->1

        >>> Faction.next(Faction.WHITE), Faction.next(Faction.BLACK)
        Faction.BLACK Faction.WHITE"""
        idx = cls._get_index(now)
        if idx == 0:
            raise ValueError()
        keys = cls.keys()
        nidx = (idx+1) % len(keys)
        return cls[keys[nidx if nidx != 0 else 1]]