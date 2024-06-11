import queue
import typing
from typing import Generic, TypeVar, Any, overload
from dataclasses import dataclass

from typing_extensions import Self
import treelib

_VT = TypeVar("_VT")
""" generic variable type """
_NT = TypeVar("_NT")
""" generic node type """

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
        self.root: _NT

    def parent(self, nid) -> _NT | None:
        return super().parent(nid)

    def children(self, nid) -> list[_NT]:
        return super().children(nid)