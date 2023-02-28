import copy
import logging
import sys
import time
from collections import Counter

import numpy as np

from board import Board
from player import RandomPlayer, MonteCarloPlayer
from game import Game

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    b = []
    results = Counter()
    times = []
    for i in range(10):
        start = time.time()
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
            results[winner.name] += 1
        else:
            logger.info('The game ended in a draw!\n\n')

            results['draw'] += 1
        times.append((time.time() - start))

    print(results)
    print(np.mean(times))

