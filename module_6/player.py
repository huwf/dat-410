from module_6.evaluate import RandomMoveMixin, MonteCarloMoveMixin
from module_6.exceptions import IllegalMoveError
from module_6.search import MonteCarloMixin, RandomSearchMixin


class Player:
    # def __init__(self, board, name='O'):
    #     self.board = board
    #     self.name = name

    def __init__(self, name='O'):
        self.name = name

    def __repr__(self):
        return f'<Player {self.name}>'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        # TODO: Make sure to check self and other are the same class
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def place(self, game, pos=None, row=None, col=None):
        if row is not None and col is not None:  # Could be 0 so check for None
            pos = (row, col)
        game.play_turn(self, pos)
        # if self.game.board[pos]:
        #     raise IllegalMoveError(f'{pos} has already been played. Pick another square')
        # self.board[pos] = self

    def play(self, game):
        possible_moves = self.search(game)
        best_move = self.evaluate(possible_moves)
        self.place(game, best_move)


class RandomPlayer(Player, RandomMoveMixin, RandomSearchMixin):
    pass
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)


class MonteCarloPlayer(Player, MonteCarloMoveMixin, MonteCarloMixin):
    pass
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
