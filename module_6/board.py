from exceptions import IllegalMoveError


class Board:
    def __init__(self, size=3, board=None):
        self.size = size
        if board is None:
            board = [[None for _ in range(size)] for _ in range(size)]
        self.board = board
        self.empty_squares = {(i, j) for i in range(size) for j in range(size)}

    def __repr__(self):
        ret = ''
        for b in self.board:
            ret += f'{str(b)} '
        return ret

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, column = item
            return self.board[row][column]
        return self.board[item]

    def __setitem__(self, key, value):
        row, column = key
        if self[key]:
            raise IllegalMoveError(f'{key} has already been played')
        self.board[row][column] = value
        self.empty_squares.remove(key)



