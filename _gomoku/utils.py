import queue
from typing import Generic, TypeVar, Any
from dataclasses import dataclass

from typing_extensions import Self

_T = TypeVar("_T")


@dataclass
class ActionBase:
    x: int
    y: int
    faction: int
    """ 所属派系 """


class Action(ActionBase):

    @property
    def package(self):
        """ 包裹用于解包

        >>> x, y, fac = self.packaged"""

        return self.x, self.y, self.faction

    def apply(self, x: int, y: int, fac: int) -> Self:
        self.x = x
        self.y = y
        self.faction = fac
        return self

    def __call__(self, x: int, y: int, fac: int):
        return self.apply(x, y, fac)


class Queue(queue.Queue, Generic[_T]):
    """ 继承于 queue.Queue 
    - 添加了更方便的下标引用 
    - 为保证后续元素的正常添加，在已满状态继续添加会自动删除最前的元素"""

    def __getitem__(self, i: int) -> _T:
        return self.queue[i]

    def put_smartly(self, item: Any) -> None | _T:
        """ 防止程序阻塞，已满状态继续添加会自动删除最前的元素并返回 """
        first = None
        if self.qsize() == self.maxsize:
            first = self.get_nowait()
            
        self.put_nowait(item)
        return first
