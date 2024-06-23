""" 多任务命令集 """
import random
import sys
from pathlib import Path
import typing
import collections
import inspect

import click
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import _internal
import _internal.env.gomoku
import _internal.env.action
import _internal.utils
# import rich.traceback
# rich.traceback.install()
import main_group as mg


@click.group(help='五子棋命令行')
def main(): pass


mg.link_CLI_options(
    func=mg.CLI0.train_from_model,
    group_cmd=main.command(help='使用预训练模型对AI进行训练'),
    options=[
        click.option('--epoch', type=int, help='迭代次数'),
        click.option('--source', type=str, help='预训练模型的路径'),
        click.option('--target', type=str, help='模型的存放的名称'),
    ]
)


mg.link_CLI_options(
    func=mg.CLI0.train_with_empty,
    group_cmd=main.command(help='从零开始对AI进行训练'),
    options=[
        click.option('--epoch', type=int, help='迭代次数'),
        click.option('--target', type=str, help='模型的存放的名称')
    ]
)

mg.link_CLI_options(
    func=mg.CLI0.agent_self_play,
    group_cmd=main.command(help='AI自博弈')
)

mg.link_CLI_options(
    func=mg.CLI0.play_with_agent,
    group_cmd=main.command(help='与AI博弈')
)


mg.link_CLI_options(
    func=mg.CLI0.play_with_human,
    group_cmd=main.command(help='与人博弈'),
    options=[
        click.option('-s', '--chess-size', default='[6, 6]', type=tuple, help="棋盘尺寸，默认是：'[6, 6]'")
    ]
)


mg.link_CLI_options(
    func=mg.CLI0.show_frame_img,
    group_cmd=main.command(help='测试GUI的渲染效果'),
    options=[
        click.option('-c', '--coverage', default='75%', type=str, help='棋盘覆盖率，默认是：75%'),
        click.option('-p', '--save-path', default=None, help='图片存放到`path`而不直接展示，若为None则只展示而不储存'),
        click.option('-s', '--chess-size', default='[6, 6]', type=tuple, help="棋盘尺寸，默认是：'[6, 6]'")
    ])


mg.link_CLI_options(
    func=mg.CLI0.draw_Network_IO,
    group_cmd=main.command(help='绘制AI模型的输入输出分析示意图'),
    options=[
        click.option('-p', '--save-folder', default='./temp/input')
    ]
)


@main.command(help='PyQt6界面-简化命令行')
def graphic_control():
    import _internal.cmd_gui as cg
    c = cg.Group()
    c.register(mg.测试)
    c.register(mg.训练)
    c.mainloop()


if __name__ == '__main__':
    main()
