""" 多任务命令集 """
import random
from pathlib import Path
import typing
import collections
import inspect

import click
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import _gomoku
import _gomoku.env
import _gomoku.utils
# import rich.traceback
# rich.traceback.install()


@click.group(help='五子棋命令行')
def main(): pass


@main.command(help='使用预训练模型对AI进行训练')
@click.option('--epoch', type=int, help='迭代次数')
@click.option('--source', type=str, help='预训练模型的路径')
@click.option('--target', type=str, help='模型的存放的名称')
def train_from_model(epoch: int, source: str, target: str):
    print(epoch, source, target)


@main.command(help='从零开始对AI进行训练')
@click.option('--epoch', type=int, help='迭代次数')
@click.option('--target', type=str, help='模型的存放的名称')
def train_with_empty(epoch, target):
    ...


@main.command(help='AI自博弈')
def agent_self_play():
    ...


@main.command(help='与AI博弈')
def play_with_agent():
    print('开始游戏')


@main.command(help='与人博弈')
@click.option('-s', '--chess-size', default='[6, 6]', type=tuple, help="棋盘尺寸，默认是：'[6, 6]'")
def play_with_human(chess_size: str):
    import pygame
    import numpy as np
    chess_size = ''.join(iter(chess_size))
    print(f'开始游戏：{chess_size=}')
    chess_size = eval(chess_size)
    players = (1, 2)
    player_idx = 0
    player = players[player_idx]
    winner = None
    with _gomoku.GomokuEnv(chess_size, 'human') as env:
        runGame = True
        while True:
            env._render_gui('human')

            mx, my = pygame.mouse.get_pos()
            i = (mx - _gomoku.env.LX)/_gomoku.env.UINT
            j = (my - _gomoku.env.LY)/_gomoku.env.UINT
            # 四舍五入
            i = round(i)
            j = round(j)
            # 限制
            i = max(0, min(i, 5))
            j = max(0, min(j, 5))
            pygame.draw.circle(env.window_surface,
                               (128, 0, 0),
                               (_gomoku.env.LX+i*_gomoku.env.UINT, _gomoku.env.LY+j*_gomoku.env.UINT),
                               5)

            for event in pygame.event.get():  # 从Pygame的事件队列中取出事件，并从队列中删除该事件
                match event.type:
                    case pygame.MOUSEBUTTONDOWN:
                        done = env.step(_gomoku.env.Action(i, j, player), False)
                        if not done:
                            continue  # 不能覆盖已有的棋子

                        player_idx += 1
                        player_idx %= len(players)
                        player = players[player_idx]

                        # 检查游戏是否结束
                        is_end, winner = env.is_end()
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


def cpm(r: float, coverage: float):
    if 0 < r < coverage:
        return 1
    elif coverage < r < coverage*2:
        return 2
    else:
        return 0


@main.command(help='测试GUI的渲染效果')
@click.option('-c', '--coverage', default='75%', type=str, help='棋盘覆盖率，默认是：75%')
@click.option('-p', '--save-path', default=None, help='图片存放到`path`而不直接展示，若为None则只展示而不储存')
def show_frame_img(coverage: str, save_path: str):
    size = (19, 19)

    coverage = float(coverage[:-1])/100
    coverage /= 2  # 防止 1.0 取小数获得的百分比是 0.0
    coverage -= int(coverage)  # 只保留小数部分

    with _gomoku.GomokuEnv(size, render_mode='rgb_array') as env:
        for i in range(size[0]):
            for j in range(size[1]):
                r = random.random()
                env.step(_gomoku.utils.Action(i, j, cpm(r, coverage)))
        arr = env.render()
        img = Image.fromarray(arr)

    if save_path is None:
        img.show()
    else:
        img.save(save_path)
        print(f'Image rendered has been saved in {Path(save_path).absolute()}')


@main.command(help='绘制AI模型的输入输出分析示意图')
@click.option('-p', '--save-folder', default='./temp/input')
def draw_Network_IO(save_folder: str):
    size = (10, 10)
    folder = Path(save_folder)

    with _gomoku.GomokuEnv(size, render_mode='rgb_array') as env0:
        for i in range(size[0]):
            for j in range(size[1]):
                r = random.random()
                env0.step(_gomoku.utils.Action(i, j, cpm(r, 0.2)), False)
        env0.step(_gomoku.utils.Action(5, 5, 1), False)

    img0 = Image.fromarray(env0.render())
    img0.save(folder / '0.png')

    with (_gomoku.GomokuEnv(size, render_mode='rgb_array') as env1,
          _gomoku.GomokuEnv(size, render_mode='rgb_array') as env2):
        env1.chessboard[env0.chessboard == 1] = 1
        env2.chessboard[env0.chessboard == 2] = 2

        arr1 = env1.render()
        arr2 = env2.render()

    Image.fromarray(arr1).save(folder / '1.png')
    Image.fromarray(arr2).save(folder / '2.png')

    with env1:
        env1.reset()
        env1.step(env0.last_action)
    Image.fromarray(env1.render()).save(folder / '3.png')

    with env1:
        env1.chessboard[:] = 1
    Image.fromarray(env1.render()).save(folder / 'final.png')

    pb = np.random.random(env1.chessboard.shape)
    env1.render_prob(pb, [75, 0, 0])
    Image.fromarray(env1.rendered_result('rgb_array')).save(folder / 'prob.png')


if __name__ == '__main__':
    main()
