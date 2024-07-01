import torch
from dataclasses import dataclass


from . import abc, typing, contain


@dataclass
class KernalData:
    x: torch.Tensor
    y: torch.Tensor


class Kernel(KernalData):
    """ 计算核，用于从棋盘中计算是否胜利 """

    def __init__(self, limit: int = 2) -> None:
        """ limit 确定范围大小【n-limit,n+limit】 """
        super().__init__(*self.get_kernel(limit))

    def peek_with(self, mat: torch.Tensor, x_vec: torch.Tensor, y_vec: torch.Tensor):
        """ 首先由 x vec(最后落棋点) 和 y vec(最后落棋点) 获取 下标矩阵(最后落棋点的[横竖对角])，然后使用下标矩阵获取mat中指定位置的数值 

        >>> arr = torch.arange(100).reshape([10,10]).cpu()
            mat = torch.zeros([2,10,10])
            mat[0]=arr
            mat[1]=-arr

        >>> px,py = torch.tensor([[4,6],[4,6]])
            px,py = px.reshape([-1,1,1]),py.reshape([-1,1,1])
        (tensor([[[4]], [[6]]]), tensor([[[4]], [[6]]]),

        >>> k = Kernel()
            k.peek_with(mat,px,py)
        tensor([[[ 24.,  34.,  44.,  54.,  64.],
                [ 42.,  43.,  44.,  45.,  46.],
                [ 22.,  33.,  44.,  55.,  66.],
                [ 66.,  55.,  44.,  33.,  22.]],
                [[-46., -56., -66., -76., -86.],
                [-64., -65., -66., -67., -68.],
                [-44., -55., -66., -77., -88.],
                [-88., -77., -66., -55., -44.]]])"""
        X = x_vec + self.x  # [n, 4, 5]
        Y = y_vec + self.y  # [n, 4, 5]
        C = torch.arange(X.shape[0], dtype=torch.int).reshape([-1, 1, 1])  # [n, 1, 1]
        res = mat[C, X, Y]  # [n, 4, 5]
        return res

    @staticmethod
    def get_kernel(n: int):
        """ 返回三维张量 """
        l = 2*n+1
        va = torch.tensor(range(-n, n+1), dtype=torch.int)
        co = torch.tensor([0]*l, dtype=torch.int)

        x = torch.zeros([4, l], dtype=torch.int)
        y = torch.zeros([4, l], dtype=torch.int)

        # 行变，列常。取横向
        x[0] = va
        y[0] = co
        # 行常，列变。取竖向
        x[1] = co
        y[1] = va
        # 行变，列变。取正对角
        x[2] = va
        y[2] = va
        # -行变，-列变。取副对角
        x[3] = -x[2]
        y[3] = y[2]

        x = x.reshape([1, 4, l])
        y = y.reshape([1, 4, l])

        return x, y




