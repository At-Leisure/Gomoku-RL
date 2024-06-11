import queue
import typing
from typing import Generic, TypeVar, Any, overload
from dataclasses import dataclass

from typing_extensions import Self
import treelib

@dataclass
class ActionBase:
    x: int = -1
    y: int = -1
    faction: int = -1
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