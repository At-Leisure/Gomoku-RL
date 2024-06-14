import torch
from typing import Generic, TypeVar, final
from torch.types import _bool, _device, _int
from typing_extensions import Self

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
""" Data Type """


class DeviceType:
    device: torch.device


@final
class CPU(DeviceType):
    device = torch.device('cpu')


@final
class CUDA(DeviceType):
    def __init__(self, idx: int = 0) -> None:
        self.device = f'cuda:{idx}'


DV = TypeVar('DV', torch.device, None,
             CPU,  CUDA)
""" Device Type """


class Tensor(Generic[DT, DV], torch.Tensor):

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> 'Tensor[Bool, CPU]':
        return super().cpu(memory_format)

    def cuda(self, device: torch.device | int | str | None = None, non_blocking: bool = False,
             memory_format: torch.memory_format = torch.preserve_format) -> 'Tensor[Bool, CUDA]':
        return super().cuda(device, non_blocking, memory_format)


def tensor(data, datatype: DT = None, device: DV = None, requires_grad=False, pin_memory=False) -> Tensor[DT, DV]:
    if isinstance(datatype, None | torch.dtype):
        dt = datatype
    elif datatype.__class__.__base__ is DataType:
        dt = datatype.dtype
    else:
        raise TypeError()

    if isinstance(device, None | torch.device):
        dv = device
    elif device.__class__.__base__ is DeviceType:
        dv = device.device
    else:
        raise TypeError()

    return torch.tensor(data, dtype=dt, device=dv, requires_grad=requires_grad, pin_memory=pin_memory)


if __name__ == '__main__':
    temp = Tensor[Bool, CUDA]
    temp = tensor(1, UInt8()).cpu()
    temp = tensor(1, torch.uint8)
    temp = tensor(1, None)
    temp = tensor(1, UInt32(), CPU()).cuda()
    """ (variable) temp: Tensor[Bool, CUDA] """
    temp = tensor(1, Bool(), CUDA()).cpu()
    """ (variable) temp: tuple[Tensor[Bool, CPU] """
