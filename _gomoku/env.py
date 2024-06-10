import os
import copy
import random
import time
import typing
from operator import itemgetter
from collections import defaultdict, deque
from typing import Literal, Any, overload
from dataclasses import dataclass
import colorsys

import pygame.freetype
from typing_extensions import Self

import gym.error
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import gym
import gym.error
from gym import logger, spaces
from gym.utils import seeding

from . import utils as _utils
from .utils import Action


__all__ = ['GomokuEnv']


RenderModesList = Literal['human', 'rgb_array', 'ansi']


UINT = 100  # 距离单元
LX = LY = UINT*3//2  # 边界距离
PR = UINT//2  # 棋子半径
FS = UINT//3  # 文字大小
FK = UINT*3//4  # 文字距离棋盘
LW = 2  # 线宽
BW = 5  # 边框宽度


COLORS = {
    'bg': (223, 191, 120),
    'chess': (217, 169, 93),
    'line': (75, 75, 75),
    'agent': {
        1: (0, 0, 0),
        2: (255, 255, 255)
    }
}


class GomokuEnv(gym.Env):
    """ 棋盘 环境 
    五子棋的环境是按照标准gym环境构建的，棋盘宽x高，先在横线、直线或斜对角线上形成n子连线的玩家获胜。
状态空间为[4,棋盘宽，棋盘高]，四个维度分别为当前视角下的位置，对手位置，上次位置以及轮次。"""

    metadata = {'render_modes': typing.get_args(RenderModesList)}
    CHESS_NULL = 0
    """ 空棋的值 """

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.chessboard = np.zeros(self.board_size, dtype=np.uint8)
        self.actions: _utils.Queue[Action] = _utils.Queue(1)
        return super().reset(seed=seed, options=options)

    def __init__(self,
                 board_size: tuple[int, int] = (6, 6),
                 render_mode: RenderModesList = None,):
        """ 
        ## Parameters
        - `board_size` 棋盘尺寸
        - `render_mode` 渲染模式"""

        super().__init__()
        self.render_mode: RenderModesList | None = render_mode
        self.board_size = np.array(board_size)
        # self.action_space = spaces.Discrete(self.board_wid*self.board_hei)
        # self.observation_space = spaces.Box()
        self.reset()

        # 检测可视环境下pygame的安装
        if self.render_mode in ['human', 'rgb_array']:
            try:
                import pygame
            except ImportError:
                raise gym.error.DependencyNotInstalled(
                    "pygame is not installed, run `pip install pygame`")
        self.window_surface = None
        self.window_size = (self.board_size-1)*UINT + 2*np.array([LX, LY])

        import pygame
        if pygame.get_init() == False:
            pygame.init()
        if self.render_mode in ["human", "rgb_array"]:
            self.window_font = pygame.font.SysFont(['Consolas'], FS)
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("None")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

    @property
    def last_action(self):
        """ 从队列中获取最后一个加入的元素 """
        return self.actions[-1]

    @property
    def board_wid(self): return self.board_size[0]
    @property
    def board_hei(self): return self.board_size[1]

    @overload
    def step(self, action: Action, forcely: bool) -> bool: pass
    @overload
    def step(self, action: Action) -> tuple[np.ndarray, int, bool]: pass

    def step(self, action: Action, forcely=None):
        """ 更新环境

        ## Parameters
        - `action` 行为对象
        - `forcely` 强制覆盖，不论是否已经有非空值存在。只在人类操纵时用到。
        """

        if isinstance(forcely, bool):
            if action.faction == self.CHESS_NULL:
                return False

            if self.chessboard[action.x, action.y] != self.CHESS_NULL:
                if not forcely:
                    return False

            self.chessboard[action.x, action.y] = action.faction
            self.actions.put_smartly(action)  # 使用Queue.put的话，在已满状态继续put就会卡住
            return True

        else:
            self.chessboard[action.x, action.y] = action.faction
            self.actions.put_smartly(action)  # 使用Queue.put的话，在已满状态继续put就会卡住

            terminated, winner = self.is_end()
            if terminated:
                if winner == action.faction:  # 获胜
                    reward = 100
                elif winner != action.faction:  # 失败
                    reward = -100
                elif winner == None:  # 持平
                    reward = 10
            else:
                reward = -1

            return self.observation_4D, reward, terminated

    @property
    def observation_4D(self):
        """ 状态的可观察量，一个四维矩阵。符合pytorch的张量，即[c,w,h]

        - 纬度1：派系1的棋子
        - 纬度2：派系2的棋子
        - 纬度3：历史落棋点
        - 纬度4：上个落棋的阵营"""
        ret = np.zeros((4, self.board_wid, self.board_hei))
        # 阵营1
        positions = self.chessboard == 1
        ret[0, positions] = 1.0
        # 阵营2
        positions = self.chessboard == 2
        ret[1, positions] = 1.0
        # 历史
        la = self.last_action
        ret[2, la.x, la.y] = 1.0
        # 对应阵营
        if la.faction == 2:
            ret[3, :, :] = 1.0
        return ret

    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        if self.render_mode in ['human', 'rgb_array']:
            self._render_gui(self.render_mode)
            return self.rendered_result(self.render_mode)
        else:
            raise NotImplementedError()

    def _render_ansi(self): raise NotImplementedError()

    def _render_gui(self, mode: RenderModesList):
        """ 仅渲染GUI """
        import pygame
        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        # init background
        self.window_surface.fill(COLORS['bg'])

        # chess
        for i in range(self.board_wid):
            for j in range(self.board_hei):
                pygame.draw.circle(self.window_surface,
                                   COLORS['chess'],
                                   [LX+i*UINT, LY+j*UINT],
                                   PR, 0)

        for i in range(0, self.board_wid):
            pygame.draw.line(self.window_surface,
                             COLORS['line'],
                             [LX+i*UINT, LY+0*UINT],
                             [LX+i*UINT, LY+(self.board_hei-1)*UINT],
                             LW)
            # A-Z
            text = chr(i+65)
            # max_len = len(str(self.board_wid))
            t = self.window_font.render(text, True, (0, 0, 0))
            self.window_surface.blit(t, [LX+i*UINT-len(text)/4*FS, LY-FK-FS])
            self.window_surface.blit(t, [LX+i*UINT-len(text)/4*FS, LY+(self.board_hei-1)*UINT+FK])
        for i in range(0, self.board_hei):
            pygame.draw.line(self.window_surface,
                             COLORS['line'],
                             [LX+0*UINT, LY+i*UINT],
                             [LX+(self.board_wid-1)*UINT, LY+i*UINT],
                             LW)
            # 0-9
            text = str(i+1)
            # max_len = len(str(self.board_hei))
            t = self.window_font.render(text, True, (0, 0, 0))
            self.window_surface.blit(t, [LX-FK-len(text)/2*FS, LY+i*UINT-FS/2])
            self.window_surface.blit(t, [LX+(self.board_wid-1)*UINT+FK, LY+i*UINT-FS/2])

        # bold boundage
        pygame.draw.rect(self.window_surface, COLORS['line'],
                         ((LX, LY), ((self.board_wid-1)*UINT, (self.board_hei-1)*UINT)), BW)

        # random circle
        # for i in range(self.board_wid):
        #     for j in range(self.board_hei):
        #         c = random.randint(0, 1)
        #         pygame.draw.circle(self.window_surface,
        #                            np.array((255, 255, 255))*c,
        #                            [LX+i*UINT, LY+j*UINT],
        #                            PR, 0)
        # self.chessboard[2,2]=1
        # self.chessboard[3,3]=2
        px, py = np.where(self.chessboard != 0)
        for idx in range(px.shape[0]):
            i, j = px[idx], py[idx]
            pygame.draw.circle(self.window_surface,
                               COLORS['agent'][self.chessboard[i, j]],
                               [LX+i*UINT, LY+j*UINT],
                               PR)

    def render_prob(self, probability_matrix: np.ndarray[np.float_],
                    color: tuple[int, int, int] | np.ndarray = None):
        """ 特殊形况，仅使用与绘制输出样式（渐变色）
        此时的 chessboard 的类型必须是 float，并且范围是 [0.0, 1.0]，灰色的深度代表了概率"""
        assert self.chessboard.shape == probability_matrix.shape, '内置棋盘和最为参数的概率棋盘的尺寸必须相等'
        w, h = probability_matrix.shape
        # 渐变色，线性插值
        aim_arr = np.array([255, 0, 0]) if color is None else np.array(color)
        base_arr = np.array(COLORS['bg'])
        gradient_color = base_arr + (aim_arr-base_arr) * probability_matrix.reshape([w, h, 1])
        gradient_color = gradient_color.astype(np.uint8)
        for i in range(self.board_wid):
            for j in range(self.board_hei):
                c = np.array([255, 0, 0])
                pygame.draw.circle(self.window_surface,
                                   gradient_color[i, j],
                                   [LX+i*UINT, LY+j*UINT],
                                   PR, 0)

    def rendered_result(self, mode: RenderModesList) -> np.ndarray:
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
        point_value = chessboard[x+EDGE, y+EDGE]
        arr[arr == point_value] = 1
        return arr

    def is_end(self) -> tuple[bool, int | None]:
        """ 返回游戏状态
        >>> is_end,winner = self.is_end()"""
        hyper_board = np.zeros([8+self.board_wid, 8+self.board_hei], dtype=self.chessboard.dtype)
        hyper_board[4:4+self.board_wid, 4:4+self.board_hei] = self.chessboard

        x, y, _ = self.last_action.package
        assert 0 <= x <= self.board_wid-1 and 0 <= y <= self.board_hei-1
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
            return True, self.chessboard[x, y]
        elif not np.any(self.chessboard == 0):  # 平局
            return True, None
        else:  # 未结束
            return False, None
