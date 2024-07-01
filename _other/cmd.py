from _internal.gui.group import CommandBaseClass, Group
from _internal.gui.input import *
from . import nn as _nn
from .env import *
from .game import *


class TrainCommand:

    ...


nndic = {
    'CNN': _nn.NormalCNN,
    'VGG': _nn.NormalVGG
}

board_width = 6        # 棋盘宽
board_height = 6       # 棋盘高
n_in_row = 4           # 胜利需要连成线棋子
c_puct = 5             # 决定探索程度
n_playout = 100        # 每步模拟次数
learn_rate = 0.002     # 学习率
lr_multiplier = 1.0    # 基于KL的自适应学习率调整
temperature = 1.0      # 温度参数
noise_eps = 0.75       # 噪声参数
dirichlet_alpha = 0.3  # dirichlet系数
buffer_size = 5000     # buffer大小
train_batch_size = 128  # batchsize大小
update_epochs = 5      # 多少个epoch更新一次
kl_coeff = 0.02        # kl系数
checkpoint_freq = 20   # 模型保存频率
mcts_infer = 200       # 纯mcts推理时间
restore_model = None   # 是否加载预训练模型
game_batch_num = 400      # 训练步数
model_path = "./models/vgg/v1"         # 模型保存路径


class 综合v0(CommandBaseClass):

    @staticmethod
    def 训练模型(棋盘尺寸: TextEditor = '[6, 6]',
             网络名称: ComboEditor = ['CNN', 'VGG'],
             模型文件: FileGetter = '',
             缓冲大小: UIntEditor = 5000,
             推理时间: UIntEditor = 200,
             KL学习率: FloatEditor = 1.0,
             单批数量: UIntEditor = 5000,
             更新周期: UIntEditor = 5,
             KL系数: FloatEditor = 0.2,
             模型学习率: FloatEditor = 0.002,
             训练步数: UIntEditor = 400,
             保存目录: FileGetter = '',
             检查周期: UIntEditor = 20,
             ______: SeparateLine = '┈┄'*20,
             单步模拟次数: UIntEditor = 100,
             探索程度系数: FloatEditor = 5,
             探索温度参数: FloatEditor = 1.0,
             探索噪声参数: FloatEditor = 0.75,
             迪利克雷系数: FloatEditor = 0.3):
        """ 设置训练的参数，保存训练的一系列文件 """
        w, h = eval(棋盘尺寸)
        Module = partial(nndic[网络名称], w, h)
        env = GomokuEnv(6, 6, 4)
        tpl = TrainPipeline(env,
                            Module,
                            模型文件,
                            缓冲大小,
                            推理时间,
                            KL学习率,
                            单批数量,
                            更新周期,
                            KL系数,
                            模型学习率,
                            训练步数,
                            保存目录,
                            检查周期,
                            #
                            单步模拟次数,
                            探索程度系数,
                            探索噪声参数,
                            迪利克雷系数,
                            探索温度参数)
        tpl.run()

    @staticmethod
    def 测试模型param(被测网络: ComboEditor = ['CNN', 'VGG'],
                  被测对象: FileGetter = '',
                  标准网络: ComboEditor = ['CNN', 'VGG'],
                  测试标准: FileGetter = ''):
        """ 测试模型
        ### 参数
        1. `被测对象` 模型文件的路径
        1. `测试标准` 如果是模型文件的路径，则使用模型进行测试；如果是空，则使用随机策略的MCTS进行测试"""
        print([被测网络, 被测对象, 标准网络, 测试标准])
        from tqdm import tqdm

        # a = []
        # # 加载模型
        # mpath = "./models/cnn/v1/best_model.pt"
        # best_policy = PolicyValueNet(model_file=mpath)
        # # 两个AI对打
        # ai_player = MCTSPlayer(best_policy.policy_value_fn)
        # mcts_player = MCTS_Pure()
        # for i in tqdm(range(100)):
        #     # 初始化棋盘
        #     board = GomokuEnv()
        #     game = Game(board)
        #     # 开始对打
        #     winner = game.start_play(ai_player, mcts_player, start_player=0)

        #     a.append(True if winner == 0 else False)
        # for i in tqdm(range(100)):
        #     # 初始化棋盘
        #     board = GomokuEnv()
        #     game = Game(board)
        #     # 开始对打
        #     winner = game.start_play(ai_player, mcts_player, start_player=1)

        #     a.append(True if winner == 0 else False)

        # print(mpath, '的胜率为', len([ai for ai in a if ai == True])/len(a))
