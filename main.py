""" 多任务命令集 """

import click
import _gomoku
import _gomoku.env


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
def play_with_human():
    import pygame
    import numpy as np
    print('开始游戏')
    players = (1, 2)
    player_idx = 0
    player = players[player_idx]
    with _gomoku.GomokuEnv((6, 6), 'human') as env:
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
                               (128, 128, 128),
                               (_gomoku.env.LX+i*_gomoku.env.UINT, _gomoku.env.LY+j*_gomoku.env.UINT),
                               5)

            for event in pygame.event.get():  # 从Pygame的事件队列中取出事件，并从队列中删除该事件
                match event.type:
                    case pygame.MOUSEBUTTONDOWN:
                        # TODO 使用 env.step 更新，而不是直接动内部矩阵
                        if env.chessboard[i, j] != 0:
                            continue  # 不能覆盖已有的棋子
                        env.chessboard[i, j] = player
                        env.last_point = np.array((i, j))
                        player_idx += 1
                        player_idx %= len(players)
                        player = players[player_idx]

                        # 检查游戏是否结束
                        is_end, winner = env.is_end()
                        if is_end and winner is not None:
                            print(f'玩家{winner}获胜')
                        elif is_end and winner is None:
                            print(f'平局')

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

    print('结束游戏')

    # while True:
    #     is_end, winner = env.is_end()
    #     if is_end:
    #         break
    #     else:
    #         player_idx += 1
    #         player_idx %= len(players)
    #         player = players[player_idx]
    #         print(player)


@main.command()
def test_render():
    import matplotlib.pyplot as plt
    with _gomoku.GomokuEnv(render_mode='rgb_array') as env:
        env.render()
        a = env.render()
        plt.imshow(a)
        plt.show()


if __name__ == '__main__':
    main()
