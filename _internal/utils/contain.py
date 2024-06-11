import queue
import typing
from typing import Generic, TypeVar, Any, overload
from dataclasses import dataclass

from typing_extensions import Self
import treelib

_VT = TypeVar("_VT")
""" generic variable type """

class Queue(queue.Queue, Generic[_VT]):
    """ 继承于 queue.Queue 
    - 添加了更方便的下标引用 
    - 为保证后续元素的正常添加，在已满状态继续添加会自动删除最前的元素"""

    def __getitem__(self, i: int) -> _VT:
        return self.queue[i]

    def put_smartly(self, item: Any) -> None | _VT:
        """ 防止程序阻塞，已满状态继续添加会自动删除最前的元素并返回 """
        first = None
        if self.qsize() == self.maxsize:
            first = self.get_nowait()

        self.put_nowait(item)
        return first