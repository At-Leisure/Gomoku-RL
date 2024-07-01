import torch

from ..utils import Kernel


class GameKernel(Kernel):
    """ 用于检测游戏胜利
    >>> gk = GameKernel(2, 5, 5, 4, torch.int)
    >>> gk.matrix = torch.tensor([
            [[0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],],
            [[1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1],]
        ])
    >>> px,py = torch.tensor([[3,2],[2,2]])
        px,py = px.reshape([-1,1,1]),py.reshape([-1,1,1])
    >>> gk.detect(px,py)
    tensor([True, True])"""

    def __init__(self, c, w, h, edge: int = 4, dtype: torch.dtype | None = None) -> None:
        """ 检测核

        ### Parameters

        - `c` 通道数
        - `w` 宽度
        - `h` 高度
        - `edge` 额外边缘距离
        - `dtype` 数据类型"""
        super().__init__(edge)
        dtype = dtype if not dtype is None else torch.int
        self.edge = edge  # 边界距离
        self.c, self.w, self.h = c, w, h
        self.hyper_matrix = torch.zeros([c, w+edge*2, h+edge*2], dtype=dtype)
        self.linear_kernel = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1]
        ], dtype=dtype)

    @property
    def matrix(self):
        return self.hyper_matrix[:, self.edge:self.w+self.edge, self.edge:self.h+self.edge]

    @matrix.setter
    def matrix(self, mat: torch.Tensor):
        self.hyper_matrix[:, self.edge:self.w+self.edge, self.edge:self.h+self.edge] = mat

    def detect(self, c_vec: torch.Tensor, x_vec: torch.Tensor, y_vec: torch.Tensor):
        x = x_vec + self.edge
        y = y_vec + self.edge
        sample = self.peek_with(self.hyper_matrix, x, y)
        # 归一消异
        equal_value = self.matrix[c_vec, x_vec, y_vec]
        equal_idx = (sample == equal_value)
        equal_idx[equal_value == 0] = False
        sample[equal_idx] = 1
        sample[~equal_idx] = 0
        # print(sample)
        sample = sample.reshape([sample.shape[0], 4, 1, 9])
        # print(f'{equal_value=}')
        res = sample * self.linear_kernel
        res = res.sum(dim=-1).reshape(res.shape[0], -1)
        # print(f'{res=}')
        return torch.any(res == 5, dim=-1)

    @property
    def traversal(self) -> torch.Tensor:
        """ 遍历matrix中的每个点，此函数通常用于检测核的测试 """
        res = torch.zeros(self.matrix.shape, dtype=torch.bool)

        for c in range(res.shape[0]):
            for x in range(res.shape[1]):
                for y in range(res.shape[2]):
                    res[c, x, y] = self.detect(torch.tensor(c), torch.tensor(x), torch.tensor(y))
        return res
