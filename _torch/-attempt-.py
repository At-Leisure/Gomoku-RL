""" 对 pytorch 使用体验做了包装，使之可以被静态类型检查。大写是为了与传统命名区分

自批：花拳绣腿，实用性零 """

import typing
import enum
from typing import Generic, TypeVar, Any, overload, Callable, Optional
from dataclasses import dataclass
from functools import wraps

from typing_extensions import Self
import treelib
import torch

_VT = TypeVar("_VT")


class SingletonMeta(type):
    """ TODO 实现单例元类，简化各数据类型的定义，基于此元类的所有类都是单例模式 """

    # def __new__(*agrs,**kwargs):
    #     print(*agrs,**kwargs)
    #     return super().__init__(*agrs,**kwargs)

    def __init__(self, *args, **kwargs):
        # print("__init__", self, *args, **kwargs)
        self.__instance = None
        super(SingletonMeta, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        # print("__call__")
        if self.__instance is None:
            self.__instance = super(SingletonMeta, self).__call__(*args, **kwargs)
        return self.__instance


class Tensor(Generic[_VT]):
    pass


class _DateType(metaclass=SingletonMeta):
    dtype: torch.dtype


@typing.final
class _Unknow(_DateType):
    dtype = None


@typing.final
class _Bool(_DateType):
    dtype = torch.bool

# -----------------------------------------------


@typing.final
class _UInt8(_DateType):
    dtype = torch.uint8


@typing.final
class _UInt16(_DateType):
    dtype = torch.uint16


@typing.final
class _UInt32(_DateType):
    dtype = torch.uint32


@typing.final
class _UInt64(_DateType):
    dtype = torch.uint64


# -----------------------------------------------
@typing.final
class _Int(_DateType):
    dtype = torch.int


@typing.final
class _Int8(_DateType):
    dtype = torch.int8


@typing.final
class _Int16(_DateType):
    dtype = torch.int16


@typing.final
class _Int32(_DateType):
    dtype = torch.int32


@typing.final
class _Int64(_DateType):
    dtype = torch.int64


# -----------------------------------------------
@typing.final
class _Float(_DateType):
    dtype = torch.float


@typing.final
class _Float16(_DateType):
    dtype = torch.float16


@typing.final
class _Float32(_DateType):
    dtype = torch.float32


@typing.final
class _Float64(_DateType):
    dtype = torch.float64


@typing.final
class _Double(_DateType):
    dtype = torch.double


none = _Unknow()
bool_ = _Bool()
# -----------------
uint8 = _UInt8()
uint16 = _UInt16()
uint32 = _UInt32()
uint64 = _UInt64()
# ----------------
int_ = _Int()
int8 = _Int8()
int16 = _Int16()
int32 = _Int32()
int64 = _Int64()
# ----------------
float_ = _Float()
float16 = _Float16()
float32 = _Float32()
float64 = _Float64()
double = _Double()

_DT = TypeVar("_DT",
              _Unknow,
              _Bool,
              _UInt8,
              _UInt16,
              _UInt32,
              _UInt64,
              _Int,
              _Int8,
              _Int16,
              _Int32,
              _Int64,
              _Float,
              _Float16,
              _Float32,
              _Float64,
              _Double)
""" generic data type """


class _DataSeries(Generic[_DT]):

    types = {}

    def __new__(cls, dt: _DT | type) -> Self:
        """ 如果类型已经被注册，那就返回注册过的对象；否则进行注册并返回 
        如果类型是 type 即表示是类，否则是类的实例"""

        if isinstance(dt, type):  # 如果传入一个类对象
            base = dt.__base__
            # print(f'class._base_: {base}')
        else:  # 传入的是类实例
            base = dt.__class__.__base__
            # print(f'instance._base_: {base}')

        if dt in cls.types.keys():
            return cls.types[dt]
        else:
            obj = cls.types[dt] = super().__new__(cls)
            return obj

    def __init__(self, datatype: _DT) -> None:
        # print(datatype.dtype)
        # print(datatype.__class__)
        # assert _DateType in datatype.__class__.__bases__
        self.dt: _DateType = datatype
        super().__init__()

    def tensor(self, data, device: torch.device | str = None,
               requires_grad: bool = False, pin_memory: bool = False) -> Tensor[_DT]:
        print(self.dt.dtype)
        return torch.tensor(data=data, dtype=self.dt.dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory)


@typing.final
class TorchNarrowTensor:
    """ （单例模式）Pytorch 中被严格明确类型的 Tensor，使用方法和torch相同。

    不改变内部规则，仅对使用体验做了包装，使之可以被静态类型检查 

    >>> Torch[uint8].tensor([1.2])
    tensor([1], dtype=torch.uint8)"""

    instance = None

    def __new__(cls) -> Self:
        if cls.instance is None:
            cls.instance = ins = super().__new__(cls)
            return ins
        else:
            return cls.instance

    def __getitem__(self, datatype: _DT):  # datatype:_VT 必须是泛型才能静态类型提示
        return _DataSeries(datatype)


TNT = Torch = TorchNarrowTensor()
""" 对 pytorch 使用体验做了包装，使之可以被静态类型检查。大写是为了与传统命名区分

>>> Torch[uint8].tensor([1.2])
tensor([1], dtype=torch.uint8)

>>> Torch[none].tensor([1.2])
tensor([1], dtype=torch.uint8)"""

if __name__ == '__main__':
    Torch[uint8].tensor([1.2])
