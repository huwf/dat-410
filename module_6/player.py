from module_6.evaluate import RandomMoveMixin
from module_6.exceptions import IllegalMoveError
from module_6.search import MonteCarloMixin, RandomSearchMixin


class Player:
    def __init__(self, board, name='O'):
        self.board = board
        self.name = name

    def __repr__(self):
        return f'<Player {self.name}>'

    def __str__(self):
        return self.__repr__()

    def place(self, pos=None, row=None, col=None):
        if row is not None and col is not None:  # Could be 0 so check for None
            pos = (row, col)
        elif not pos:
            raise IllegalMoveError('')

        if self.board[pos]:
            raise IllegalMoveError(f'{pos} has already been played. Pick another square')
        self.board[pos] = self

    def play(self):
        possible_moves = self.search(self.board)
        best_move = self.evaluate(possible_moves)
        self.place(best_move)


class RandomPlayer(Player, RandomMoveMixin, RandomSearchMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

