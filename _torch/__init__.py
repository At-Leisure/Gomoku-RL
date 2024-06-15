""" 对 pytorch 使用体验做了包装，对 Tensor 进行了继承。

使之可以被静态类型检查。实质没有变化

TODO 在项目完成后把所有tensor转换成本地的tensor形式
TODO 在此刻后的所有tensor都使用本地定义的类似静态形式

"""

from .review import *
from .const import *

""" 🦄 refactor(tensor): 停止费力且低效的静态类型适配

删除了 Tensor.to 的适配，返回子类对象，结束适配工作！

评价：函数繁多，需求微有，表面工作，杂糅时间。 """