import sys
import time

from module_6.board import Board
from module_6.player import RandomPlayer, MonteCarloPlayer
from module_6.game import Game

if __name__ == '__main__':
    b = []

    board = Board(3)
    p1 = RandomPlayer('O')
    p2 = RandomPlayer('X')
    game = Game(p1, p2, board)
    p1 = MonteCarloPlayer('O')
    p2 = MonteCarloPlayer('X')
    game = Game(p1, p2, board, next_player=p1)
    #
    while not game.is_finished:
        game.next_player.play(game)

    if winner := game.winner:
        print(f'{winner} Won!')
    else:
        print('The game ended in a draw!')
