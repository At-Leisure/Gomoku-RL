import os
import copy
import random
import time
from operator import itemgetter
from collections import defaultdict, deque
from typing import Literal, Any

import gym.error
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import gym
import gym.error
from gym import logger, spaces, utils
from gym.utils import seeding
# import gym
# from gym.spaces import Box, Discrete


class EndState:
    Tie = 0
    NoTie = 1


__all__ = ['GomokuEnv']


def _items(x) -> tuple[str | Any]:
    """ get all the elements of Literal instance."""
    return x.__args__


RenderModesList = Literal['human', 'rgb_array', 'ansi']

UINT = 30  # 距离单元
LX = LY = UINT//2  # 边界距离
PR = 10  # 棋子半径


point_color = (None, (0, 0, 0), (255, 255, 255))

colors = {
    'bg': (228, 206, 161),
    'line': (0, 0, 0)
}


class GomokuEnv(gym.Env):
    """ 棋盘 环境 """

    metadata = {'render_modes': _items(RenderModesList)}

    def __init__(self,
                 board_size: tuple[int, int] = (6, 6),
                 render_mode: RenderModesList = None,):
        super().__init__()
        self.render_mode: RenderModesList | None = render_mode
        self.board_size = np.array(board_size)
        self.last_point = np.array([-1, -1])
        self.chessboard = np.zeros(board_size, dtype=np.uint8)
        self.action_space = spaces.Discrete(self.board_wid*self.board_hei)
        # self.observation_space = spaces.Box()

        # 检测可视环境下pygame的安装
        if self.render_mode in ['human', 'rgb_array']:
            try:
                import pygame as pg
            except ImportError:
                raise gym.error.DependencyNotInstalled(
                    "pygame is not installed, run `pip install pygame`")
        self.window_surface = None
        self.window_size = self.board_size*UINT

        import pygame
        if pygame.get_init() == False:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("None")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

    @property
    def board_wid(self): return self.board_size[0]
    @property
    def board_hei(self): return self.board_size[1]

    def step(self, action: Any):
        return super().step(action)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.chessboard = np.zeros(self.board_size, dtype=np.uint8)
        self.last_point = np.array([-1, -1])
        return super().reset(seed=seed, options=options)

    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        if self.render_mode in ['human', 'rgb_array']:
            self._render_gui(self.render_mode)
            return self._render_result(self.render_mode)
        else:
            raise NotImplementedError()

    def _render_ansi(self): pass

    def _render_gui(self, mode: RenderModesList):
        """ 仅渲染GUI """
        import pygame
        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        # init background
        self.window_surface.fill(colors['bg'])
        # self.game_surface.fill(colors['bg'])

        for i in range(0, self.board_wid):
            pygame.draw.line(self.window_surface,
                             colors['line'],
                             [LX+i*UINT, LY+0*UINT],
                             [LX+i*UINT, LY+(self.board_hei-1)*UINT])
        for i in range(0, self.board_hei):
            pygame.draw.line(self.window_surface,
                             colors['line'],
                             [LX+0*UINT, LY+i*UINT],
                             [LX+(self.board_wid-1)*UINT, LY+i*UINT])

        # random circle
        # for i in range(self.board_wid):
        #     for j in range(self.board_hei):
        #         c = random.randint(0, 1)
        #         pygame.draw.circle(self.window_surface,
        #                            np.array((255, 255, 255))*c,
        #                            [LX+i*UINT, LY+j*UINT],
        #                            5, 0)
        # self.chessboard[2,2]=1
        # self.chessboard[3,3]=2
        px, py = np.where(self.chessboard != 0)
        for idx in range(px.shape[0]):
            i, j = px[idx], py[idx]
            pygame.draw.circle(self.window_surface,
                               point_color[self.chessboard[i, j]],
                               [LX+i*UINT, LY+j*UINT],
                               PR)

    def _render_result(self, mode: RenderModesList):
        """ 返回GUI的渲染结果 """
        import pygame
        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            # self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def close(self):
        return super().close()

    def _getPointAround(self, x: int, y: int, chessboard: np.ndarray) -> np.ndarray:
        """ 获取指定点的四个方向的所有棋子
        ## Params
        - `x` 坐标x
        - `y` 坐标y
        ## Return
        返回一个形状为[4，9]的矩阵
        ## Note
        - 如果是中间的棋子，就直接通过索引计算
        - 如果是边缘的棋子，为了避免索引越界，对不能获得的棋子直接置零"""
        arr = np.zeros([4, 9], dtype=np.int32)
        EDGE = 4
        # 横向
        mat = chessboard[EDGE+x-4:EDGE+x+5, EDGE+y-4:EDGE+y+5]
        mat[mat != chessboard[EDGE+x, EDGE+y]] = 0
        arr[0] = chessboard[EDGE+x-4:EDGE+x+5, EDGE+y]
        arr[1] = chessboard[EDGE+x, EDGE+y-4:EDGE+y+5]
        arr[2] = np.diagonal(mat)  # 使用np.diagonal提取正对角线元素
        arr[3] = np.diagonal(np.fliplr(mat))  # 使用np.diagonal提取正对角线元素
        # 归一
        point_value = chessboard[x+EDGE,y+EDGE]
        arr[arr == point_value]=1
        return arr

    def is_end(self) -> tuple[bool, int | None]:
        """ 返回游戏状态
        >>> is_end,winner = self.is_end()"""
        hyper_board = np.zeros([8+self.board_wid, 8+self.board_hei], dtype=self.chessboard.dtype)
        hyper_board[4:4+self.board_wid, 4:4+self.board_hei] = self.chessboard
        assert np.all(self.last_point >= 0)
        x, y = self.last_point
        arr = self._getPointAround(x, y, hyper_board)

        kernel = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        kernel = np.array([kernel[i:i+9] for i in range(5)])
        """[[0 0 0 0 1 1 1 1 1]
            [0 0 0 1 1 1 1 1 0]
            [0 0 1 1 1 1 1 0 0]
            [0 1 1 1 1 1 0 0 0]
            [1 1 1 1 1 0 0 0 0]] """
        res = arr.reshape(4, 1, 9)*kernel
        res = np.sum(res, axis=-1)
        is_win = np.any(res == 5)

        if is_win:  # 胜局
            return True, self.chessboard[self.last_point[0],self.last_point[1]]
        elif not np.any(self.chessboard == 0):  # 平局
            return True, None
        else:  # 未结束
            return False, None
