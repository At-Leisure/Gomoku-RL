from .env import *
from .nn import *

# 定义当前玩家


class CurPlayer:
    player_id = 0


# 可视化部分
class Game(object):
    def __init__(self, board, wid, hei):
        self.board = board
        self.board_width = wid
        self.board_height = hei
        self.cell_size = self.board_width - 1
        self.chess_size = 50 * self.cell_size

        self.whitex = []
        self.whitey = []
        self.blackx = []
        self.blacky = []

        # 棋盘背景色
        self.color = "#e4ce9f"
        self.colors = [[self.color] * self.cell_size for _ in range(self.cell_size)]

    def graphic(self, board, player1, player2):
        global run_i
        """Draw the board and show game info"""
        plt_fig, ax = plt.subplots(facecolor=self.color)
        ax.set_facecolor(self.color)

        # 制作棋盘
        # mytable = ax.table(cellColours=self.colors, loc='center')
        mytable = plt.table(cellColours=self.colors,
                            colWidths=[1 / self.board_width] * self.cell_size,
                            loc='center'
                            )

        ax.set_aspect('equal')

        # 网格大小
        cell_height = 1 / self.board_width
        for pos, cell in mytable.get_celld().items():
            cell.set_height(cell_height)

        mytable.auto_set_font_size(False)
        mytable.set_fontsize(self.cell_size)
        ax.set_xlim([1, self.board_width * 2 + 1])
        ax.set_ylim([self.board_height * 2 + 1, 1])
        plt.title("Gomoku")

        plt.axis('off')
        cur_player = CurPlayer()

        run_i = 0
        while True:

            # left down of mouse
            try:
                if cur_player.player_id == 1:
                    move = player1.get_action(self.board)
                    self.board.step(move)
                    x, y = self.board.move_to_location(move)
                    plt.scatter((y + 1) * 2, (x + 1) * 2, s=self.chess_size, c='white')
                    cur_player.player_id = 0
                elif cur_player.player_id == 0:
                    move = player2.get_action(self.board)
                    self.board.step(move)
                    x, y = self.board.move_to_location(move)
                    plt.scatter((y + 1) * 2, (x + 1) * 2, s=self.chess_size, c='black')
                    cur_player.player_id = 1

                end, winner = self.board.game_end()
                if end:
                    if winner != -1:
                        ax.text(x=self.board_width, y=(self.board_height + 1) * 2 + 0.1,
                                s="Game end. Winner is player {}".format(cur_player.player_id), fontsize=10,
                                color='red', weight='bold',
                                horizontalalignment='center')
                    else:
                        ax.text(x=self.board_width, y=(self.board_height + 1) * 2 + 0.1,
                                s="Game end. Tie Round".format(cur_player.player_id), fontsize=10, color='red',
                                weight='bold',
                                horizontalalignment='center')
                    run_i += 1
                    plt.gcf().savefig(f'./gif/{run_i}.png')
                    return winner
                display.display(plt.gcf())
                display.clear_output(wait=True)
            except:
                pass

            run_i += 1
            plt.gcf().savefig(f'./gif/{run_i}.png')

    def run(self, board: GomokuEnv, player1: MCTS_Pure, player2: MCTS_Pure):
        cur_player = CurPlayer()
        while True:
            # left down of mouse
            try:
                if cur_player.player_id == 1:
                    move = player1.get_action(self.board)
                    self.board.step(move)
                    cur_player.player_id = 0
                elif cur_player.player_id == 0:
                    move = player2.get_action(self.board)
                    self.board.step(move)
                    cur_player.player_id = 1

                end, winner = self.board.game_end()
                if end:
                    if winner != -1:
                        return cur_player.player_id
                    else:
                        return None
            except:
                pass

    def start_play(self, player0: MCTS_Pure, player1: MCTS_Pure, start_player=0):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.reset()
        p1, p2 = self.board.players
        player0.set_player_ind(p1)
        player1.set_player_ind(p2)
        return self.run(self.board, player0, player1)
        # self.graphic(self.board, player1, player2)
