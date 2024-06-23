from PIL import Image
import random
import click
from pathlib import Path
from typing import Callable, Any

import _internal.cmd_gui as cg


def link_CLI_options(*, func: Callable, options: list[Callable] = None, group_cmd: Callable = None):
    """ click库 不使用 @装饰器 """
    opted = func  # 开始使用cli的option装饰func
    if not options is None:
        for option in reversed(options):
            # 按照正常顺序逐步装饰cli参数
            opted = option(opted)
    if not group_cmd is None:
        # 注册到 cli group 的命令组
        group_cmd(opted)


def cpm(r: float, coverage: float):
    if 0 < r < coverage:
        return 1
    elif coverage < r < coverage*2:
        return 2
    else:
        return 0


class CLI0:

    @staticmethod
    def train_from_model(epoch: int, source: str, target: str):
        print(epoch, source, target)

    @staticmethod
    def train_with_empty(epoch, target):
        ...

    @staticmethod
    def play_with_agent():
        print('开始游戏')

    @staticmethod
    def agent_self_play():
        ...

    @staticmethod
    def play_with_human(chess_size: str):
        import pygame
        import numpy as np
        import _internal
        import _internal.env.gomoku
        import _internal.env.action

        chess_size = ''.join(iter(chess_size))
        print(f'开始游戏：{chess_size=}')
        chess_size = eval(chess_size)
        players = (1, 2)
        player_idx = 0
        player = players[player_idx]
        winner = None
        with _internal.env.gomoku.GomokuEnv(chess_size, 'human') as env:
            runGame = True
            while True:
                env._render_gui('human')

                mx, my = pygame.mouse.get_pos()
                i = (mx - _internal.env.gomoku.LX)/_internal.env.gomoku.UINT
                j = (my - _internal.env.gomoku.LY)/_internal.env.gomoku.UINT
                # 四舍五入
                i = round(i)
                j = round(j)
                # 限制
                i = max(0, min(i, 5))
                j = max(0, min(j, 5))
                pygame.draw.circle(env.window_surface,
                                   (128, 0, 0),
                                   (_internal.env.gomoku.LX+i*_internal.env.gomoku.UINT, _internal.env.gomoku.LY+j*_internal.env.gomoku.UINT),
                                   5)

                for event in pygame.event.get():  # 从Pygame的事件队列中取出事件，并从队列中删除该事件
                    match event.type:
                        case pygame.MOUSEBUTTONDOWN:
                            done = env.step(_internal.env.gomoku.Action(i, j, player), False)
                            if not done:
                                continue  # 不能覆盖已有的棋子

                            player_idx += 1
                            player_idx %= len(players)
                            player = players[player_idx]

                            # 检查游戏是否结束
                            is_end, winner = env.is_won()
                            # if is_end and winner is not None:
                            #     print(f'玩家{winner}获胜')
                            # elif is_end and winner is None:
                            #     print(f'平局')

                            if is_end:
                                runGame = False

                        case pygame.QUIT:
                            runGame = False

                if not runGame:
                    pygame.quit()
                    break
                else:
                    pygame.event.pump()
                    pygame.display.flip()
                    pygame.display.update()

        print(f'游戏结束：{f"玩家{winner}号获胜" if winner else "平局"}')

    @staticmethod
    def show_frame_img(coverage: str = '75%', save_path: str = None, chess_size: str = '[10, 10]'):
        import _internal
        import _internal.env.gomoku
        import _internal.env.action

        chess_size = ''.join(iter(chess_size))
        size = eval(chess_size)

        coverage = float(coverage[:-1])/100
        coverage /= 2  # 防止 1.0 取小数获得的百分比是 0.0
        coverage -= int(coverage)  # 只保留小数部分

        with _internal.env.gomoku.GomokuEnv(size, render_mode='rgb_array') as env:
            for i in range(size[0]):
                for j in range(size[1]):
                    r = random.random()
                    env.step(_internal.env.action.Action(i, j, cpm(r, coverage)), False)
            arr = env.render()
            img = Image.fromarray(arr)

        try:
            img.save(save_path)
            print(f'Image rendered has been saved in {Path(save_path).absolute()}')
        except:
            print(f'{save_path} 路径不存在，将不进行存储到磁盘')
            img.show()

    @staticmethod
    def draw_Network_IO(save_folder: str):
        import _internal
        import _internal.env.gomoku
        import _internal.env.action
        import numpy as np
        size = (10, 10)
        folder = Path(save_folder)
        try:
            with _internal.env.gomoku.GomokuEnv(size, render_mode='rgb_array') as env0:
                for i in range(size[0]):
                    for j in range(size[1]):
                        r = random.random()
                        env0.step(_internal.env.action.Action(i, j, cpm(r, 0.2)), False)
                env0.step(_internal.env.action.Action(5, 5, 1), False)

            img0 = Image.fromarray(env0.render())
            img0.save(folder / '0.png')

            with (_internal.env.gomoku.GomokuEnv(size, render_mode='rgb_array') as env1,
                  _internal.env.gomoku.GomokuEnv(size, render_mode='rgb_array') as env2):
                env1.chessboard[env0.chessboard == 1] = 1
                env2.chessboard[env0.chessboard == 2] = 2

                arr1 = env1.render()
                arr2 = env2.render()

            Image.fromarray(arr1).save(folder / '1.png')
            Image.fromarray(arr2).save(folder / '2.png')

            with env1:
                env1.reset()
                env1.step(env0.last_action, True)
            Image.fromarray(env1.render()).save(folder / '3.png')

            with env1:
                env1.chessboard[:] = 1
            Image.fromarray(env1.render()).save(folder / 'final.png')

            pb = np.random.random(env1.chessboard.shape)
            env1.render_prob(pb, [75, 0, 0])
            Image.fromarray(env1.rendered_result('rgb_array')).save(folder / 'prob.png')
        except FileNotFoundError as e:
            print(e, '文件夹未创建')

# ----------------------------------------------- GUI ----------------------------------------------- #


class 测试(cg.CmdBase):

    @staticmethod
    def 测试GUI可用(参数1: cg.UIntEditer = 0, 参数2: cg.IntEditer = 0,
                参数3: cg.TextEditer = '', 参数4: cg.FloatEditer = 0.,
                参数5: cg.PathEditer = ''):
        """ GUI 测试
        ### Parameters
        - `参数1` 自然数
        - `参数2` 整数类型
        - `参数3` 纯文本类型
        - `参数4` 浮点数类型
        - `参数5` 路径类型
        """
        print(参数1, 参数2, 参数3, 参数4, 参数5)

    @staticmethod
    def 展示渲染帧的效果(覆盖率: cg.TextEditer = '75%',
                 保存路径: cg.PathEditer = '',
                 棋盘尺寸: cg.TextEditer = '[10, 10]'):
        """ ### 参数
        - `覆盖率` 棋盘覆盖率，默认是：75%
        - `保存路径` 图片存放到`path`而不直接展示，若为None则只展示而不储存
        - `棋盘尺寸` 默认是：'[6, 6]'
        """
        CLI0.show_frame_img(覆盖率, 保存路径, 棋盘尺寸)
        
    @staticmethod
    def 绘制网络的数据流(保存目录: cg.PathEditer = './temp/netio'):
        """ ### 参数
        - `保存目录` 生成的图片组保存的目录
        """
        CLI0.draw_Network_IO(保存目录)


class 训练(cg.CmdBase):
    ...
