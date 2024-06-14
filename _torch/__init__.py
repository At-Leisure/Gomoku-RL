""" 对 pytorch 使用体验做了包装，使之可以被静态类型检查。 
TODO 在项目完成后把所有tensor转换成本地的tensor形式
TODO 在此刻后的所有tensor都使用本地定义的类似静态形式"""

from .basic import *
from .const import *
