import queue
from collections import deque
import typing
from typing import Generic, TypeVar, Any, overload
from dataclasses import dataclass

from typing_extensions import Self
import treelib

_VT = TypeVar("_VT")
""" generic variable type """


class seque(deque, Generic[_VT]):
    """ 单端队列(和线程无关的) single-ended queue，继承于deque双端队列，限制了双端功能"""

    def put_smartly(self, item: Any) -> None | _VT:
        """ 已满状态继续添加会自动删除最前的元素并返回 """
        ret = None
        if len(self) == self.maxlen:
            ret = self.popleft()
        self.append(item)
        return ret
    
    def __getitem__(self, key: typing.SupportsIndex) -> _VT:
        return super().__getitem__(key)

# class Queue(queue.Queue, Generic[_VT]):
#     """ 继承于 queue.Queue
#     - 添加了更方便的下标引用
#     - 为保证后续元素的正常添加，在已满状态继续添加会自动删除最前的元素"""

#     def __getitem__(self, i: int) -> _VT:
#         return self.queue[i]

#     def put_smartly(self, item: Any) -> None | _VT:
#         """ 防止程序阻塞，已满状态继续添加会自动删除最前的元素并返回 """
#         first = None
#         if self.qsize() == self.maxsize:
#             first = self.get_nowait()

#         self.put_nowait(item)
#         return first
