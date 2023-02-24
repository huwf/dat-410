import sys
import time

from module_6.board import Board
from module_6.player import RandomPlayer


if __name__ == '__main__':
    b = Board(3)
    p1 = RandomPlayer(b)
    p2 = RandomPlayer(b, 'X')
    next_player = p1
    i = 0
    while b.empty_squares:

        next_player = p1 if i % 2 == 0 else p2
        next_player.play()
        print(str(b))
        if b.winner:
            print(f'{next_player} Won!')
            sys.exit(0)
        i += 1
    print('The game ended in a draw!')
