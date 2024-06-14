import torch

from .basic import (Bool, UInt8, UInt16, UInt32, UInt64,
                    Int, Int8, Int16, Int32, Int64,
                    Float, Float16, Float32, Float64, Double,
                    #
                    CPU, CUDA,
                    #
                    Tensor, tensor)
__all__ = ['bool_', 'uint8', 'uint16', 'uint32', 'uint64', 'int_',
           'int8', 'int16', 'int32', 'int64',
           'float_', 'float16', 'float32', 'float64', 'double',
           # ------------------------------------
           'cpu', 'cuda',]

# DT = TypeVar('DT', torch.dtype, None,
#              DataType, Bool,
#              UInt8, UInt16, UInt32, UInt64,
#              Int, Int8, Int16, Int32, Int64,
#              Float, Float16, Float32, Float64,
#              Double)

# DV = TypeVar('DV', torch.device, None,
#              CPU, GPU, CUDA)
# -------------------------------------
bool_ = Bool()
#
uint8 = UInt8()
uint16 = UInt16()
uint32 = UInt32()
uint64 = UInt64()
#
int_ = Int()
int8 = Int8()
int16 = Int16()
int32 = Int32()
int64 = Int64()
#
float_ = Float()
float16 = Float16()
float32 = Float32()
float64 = Float64()
#
double = Double()
# ------------------------------------
#
cpu = CPU()
cuda = CUDA()


if __name__ == '__main__':
    temp = tensor(1, uint8)
    temp = tensor(1, torch.uint8)
    temp = tensor(1, None)
    temp = tensor(1, int32, cpu)
    temp = tensor(1, bool_, cuda)
