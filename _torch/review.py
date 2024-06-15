""" 类型审查 """

from typing import Generic, TypeVar, final, overload, Optional
from functools import singledispatch  # singledispatch 的重载没有提示功能
from typing_extensions import Self

import torch

__all__ = ['Tensor', 'tensor']


class DataType:
    dtype: torch.dtype


@final
class Bool(DataType):
    dtype = torch.bool


@final
class UInt8(DataType):
    dtype = torch.uint8


@final
class UInt16(DataType):
    dtype = torch.uint16


@final
class UInt32(DataType):
    dtype = torch.uint32


@final
class UInt64(DataType):
    dtype = torch.uint64


@final
class Int(DataType):
    dtype = torch.int


@final
class Int8(DataType):
    dtype = torch.int8


@final
class Int16(DataType):
    dtype = torch.int16


@final
class Int32(DataType):
    dtype = torch.int32


@final
class Int64(DataType):
    dtype = torch.int64


@final
class Float(DataType):
    dtype = torch.float


@final
class Float16(DataType):
    dtype = torch.float16


@final
class Float32(DataType):
    dtype = torch.float32


@final
class Float64(DataType):
    dtype = torch.float64


@final
class Double(DataType):
    dtype = torch.double


DT = TypeVar('DT', torch.dtype, None,
             Bool, UInt8, UInt16, UInt32, UInt64,
             Int, Int8, Int16, Int32, Int64,
             Float, Float16, Float32, Float64, Double)
""" Data Type at class"""
DT_ = TypeVar('DT_', torch.dtype, None,
              Bool, UInt8, UInt16, UInt32, UInt64,
              Int, Int8, Int16, Int32, Int64,
              Float, Float16, Float32, Float64, Double)
""" Data Type at method"""


class DeviceType:
    device: torch.device


@final
class CPU(DeviceType):
    device = torch.device('cpu')


@final
class CUDA(DeviceType):
    def __init__(self, idx: int = 0) -> None:
        self.device = f'cuda:{idx}'


DV = TypeVar('DV', torch.device, str, None,
             CPU,  CUDA)
""" Device Type at class """
DV_ = TypeVar('DV_', torch.device, None,
              CPU,  CUDA)
""" Device Type at method"""


class Tensor(Generic[DT, DV], torch.Tensor):
    """ 继承自 torch.Tensor，适配静态类型提示功能(目前只在定义时有效，运算后失效)。
    为了适配 DataType，DeviceType 和 Tensor.to()，必须要重新定义原有的部分函数。

    ## Note 

    #### 浅拷贝机制

    如果改变通过浅拷贝得到新对象，则源对象也会同步变化

    >>> t = torch.tensor([1, 2])
    >>> ten = Ten(t), ten[0]=10
    >>> print(t,ten)
    tensor([10, 2]) Ten([10, 2])

    通过拷贝复制函数的新对象和原对象是同一个内存

    >>> class Ten(torch.Tensor): pass # 继承并定义一个新类

    >>> t = torch.rand([1000,1000,1000],device='cuda') # 先在cuda中创建如下一个tensor，并使用 nvidia-smi 查看 GPU 的内存用量
    |======================|
    |   1807MiB /  6144MiB |
    |======================|
    >>> ten = Ten(t) # 使用新的类二次定义，观察到 GPU 的内存用量并没有变化
    |======================|
    |   2953MiB /  6144MiB |
    |======================|
    >>> t2 = torch.rand([300,1000,1000],device='cuda') # 如果在cuda中再次创建如下一个tensor，则 GPU 的内存用量会增长
    |======================|
    |   2953MiB /  6144MiB |
    |======================|
    """

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> 'Tensor[DT, CPU]':
        return Tensor(super().cpu(memory_format=memory_format))

    def cuda(self, device: torch.device | int | str | None = None, non_blocking: bool = False,
             memory_format: torch.memory_format = torch.preserve_format) -> 'Tensor[DT, CUDA]':
        return Tensor(super().cuda(device=device, non_blocking=non_blocking, memory_format=memory_format))

    def int(self) -> 'Tensor[Int32, DV]':
        return Tensor(super().int())

    def bool(self) -> 'Tensor[Bool, DV]':
        return Tensor(super().bool())

    def float(self) -> 'Tensor[Float32, DV]':
        return Tensor(super().float())


def tensor(data, datatype: DT = None, devicetype: DV = None, requires_grad=False, pin_memory=False) -> Tensor[DT, DV]:
    if isinstance(datatype, None | torch.dtype):
        dt = datatype
    elif datatype.__class__.__base__ is DataType:
        dt = datatype.dtype
    else:
        raise TypeError()

    if isinstance(devicetype, None | torch.device | str):
        dv = devicetype
    elif devicetype.__class__.__base__ is DeviceType:
        dv = devicetype.device
    else:
        raise TypeError()

    return Tensor(torch.tensor(data, dtype=dt, device=dv, requires_grad=requires_grad, pin_memory=pin_memory))


if __name__ == '__main__':
    temp = Tensor[Bool, CUDA]
    temp = tensor(1, torch.uint8)
    temp = tensor(1, UInt8()).cpu()
    temp = tensor(1, None).to(temp)
    temp = tensor(1, UInt32(), CPU()).cuda()
    """ (variable) temp: Tensor[Bool, CUDA] """
    temp = tensor(1, Bool(), CUDA()).cpu()
    """ (variable) temp: tuple[Tensor[Bool, CPU] """
