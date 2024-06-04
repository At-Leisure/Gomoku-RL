"""  """

import click


@click.group(help='五子棋命令行')
def main():
    ...


@main.command(help='开始游戏')
def play():
    print('开始游戏')


if __name__ == '__main__':
    main()
