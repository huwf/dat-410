from module_6.exceptions import IllegalMoveError


class Game:
    def __init__(self, p1, p2, board, next_player=None):
        self.p1 = p1
        self.p2 = p2
        self.board = board
        self.next_player = self.p1 if next_player is None else self.p2

    def _is_winner(self):
        size = self.board.size
        # All straight
        # down = [{self.board[j][i] for j in range(size)} for i in range(size)]
        down = [{self.board[(j, i)] for j in range(size)} for i in range(size)]
        for i in range(size):
            if all(self.board[i]) and len(set(self.board[i])) == 1:
                print(f'Winning with line {i} across')
                return self.board[(i, 0)]
            if all(down[i]) and len(down[i]) == 1:
                print(f'Winning with line {i} down')
                return down[i].pop()

        # if all(self.board[:, i]) and len(self.board[:, i]) == 1:
        #     return self.board[0, i]
        # Diagonals
        diag_l_r = {self.board[(i, i)] for i in range(size)}
        if any(diag_l_r) and len(diag_l_r) == 1:
            print('Winning with diagonal left to right')
            return diag_l_r.pop()
        diag_r_l = {(i, size - i) for i in range(size)}
        if any(diag_r_l) and len(diag_r_l) == 1:
            print('Winning with diagonal right to left')
            return diag_r_l.pop()
        return None

    @property
    def winner(self):
        return self._is_winner()

    @property
    def is_draw(self):
        return self.is_finished and not self.winner

    @property
    def is_finished(self):
        """If there is a winner or there are no squares left, the game is over"""
        return self._is_winner() or not self.board.empty_squares

    def play_turn(self, player, pos):
        assert player == self.next_player, f"It is not {player}'s turn. {self.next_player} is to play"
        self.next_player = self.p1 if self.next_player == self.p2 else self.p2
        if not self.is_illegal_move(pos):
            self.board[pos] = player
        print(self.board)

    def is_illegal_move(self, pos):
        if self.board[pos]:
            raise IllegalMoveError(f'{pos} has already been played. Pick another move')
        return False
