from _torch import *


temp: None
""" 
>>> # output
_torch.basic.Tensor[<_torch.basic.Bool object at 0x0000021637BB2DA0>, <_torch.basic.CUDA object at 0x0000021637BB3010>]
tensor(1, dtype=torch.uint8)
tensor(1, dtype=torch.uint8)
tensor(1)
tensor(1, device='cuda:0', dtype=torch.uint32)
tensor(True)
OK"""

if __name__ == '__main__':
    temp = Tensor[bool_, cuda]
    print(temp)
    temp = tensor(1, uint8).cpu()
    print(temp)
    temp = tensor(1, uint8)
    print(temp)
    temp = tensor(1, None)
    print(temp)
    temp = tensor(1, uint32, cpu).cuda()
    """ (variable) temp: Tensor[Bool, CUDA] """
    print(temp)
    temp = tensor(1, bool_, cuda).cpu()
    """ (variable) temp: tuple[Tensor[Bool, CPU] """
    print(temp)

    print('OK')
