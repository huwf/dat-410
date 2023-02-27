import numpy as np

from exceptions import IllegalMoveError


class Board:
    def __init__(self, size=3):
        self.size = size
        self.board = [[None for _ in range(size)] for _ in range(size)]
        self.empty_squares = {(i, j) for i in range(size) for j in range(size)}

    def __repr__(self):
        ret = ''
        for b in self.board:
            ret += f'{str(b)}\n'
        return ret

    def __getitem__(self, item):
        row, column = item
        return self.board[row][column]

    def __setitem__(self, key, value):
        row, column = key
        if self[key]:
            raise IllegalMoveError(f'{key} has already been played')
        self.board[row][column] = value
        self.empty_squares.remove(key)

    def _is_winner(self):
        # All straight
        down = [{self.board[j][i] for j in range(self.size)} for i in range(self.size)]
        for i in range(self.size):
            if all(self.board[i]) and len(set(self.board[i])) == 1:
                #print(f'Winning with line {i} across')
                return self[(i, 0)]
            if all(down[i]) and len(down[i]) == 1:
                #print(f'Winning with line {i} down')
                return down[i].pop()

        # if all(self.board[:, i]) and len(self.board[:, i]) == 1:
        #     return self.board[0, i]
        # Diagonals
        diag_l_r = {self[(i, i)] for i in range(self.size)}
        if any(diag_l_r) and len(diag_l_r) == 1:
            #print('Winning with diagonal left to right')
            return diag_l_r.pop()
        diag_r_l = {(i, self.size - i) for i in range(self.size)}
        if any(diag_r_l) and len(diag_r_l) == 1:
            #print('Winning with diagonal right to left')
            return diag_r_l.pop()
        return None

    @property
    def winner(self):
        return self._is_winner()

    def is_finished(self):
        """If there is a winner or there are no squares left, the game is over"""
        return self._is_winner() or not self.empty_squares


