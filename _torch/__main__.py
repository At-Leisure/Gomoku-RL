import click

from .review import *
from .const import *


@click.command()
def test():
    a = tensor(range(5), float16); print(a)
    a = tensor(range(5), uint8, cpu); print(a)
    a = tensor(range(5), devicetype=cpu).cuda().int(); print(a)
    a = tensor(range(5), devicetype=cuda).cpu().float(); print(a)
    a = tensor(range(5)).cpu().bool(); print(a)
    a = tensor(range(5), double ); print(a)


if __name__ == '__main__':
    test()
