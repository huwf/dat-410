import copy
import logging
import sys
import time

from module_6.board import Board
from module_6.player import RandomPlayer, MonteCarloPlayer
from module_6.game import Game

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    b = []

    board = Board(3)
    # p1 = RandomPlayer('O')
    # p2 = RandomPlayer('X')
    # game = Game(p1, p2, board)
    p1 = MonteCarloPlayer('O')
    p2 = MonteCarloPlayer('X')
    game = Game(p1, p2, board, next_player=p1)
    #
    moves = [copy.deepcopy(game.board)]
    while not game.is_finished:
        game.next_player.play(game)
        moves.append(copy.deepcopy(game.board))

    for m in moves:
        logger.info(m)

    if winner := game.winner:
        logger.info(f'{winner} Won!\n\n')
    else:
        logger.info('The game ended in a draw!\n\n')

