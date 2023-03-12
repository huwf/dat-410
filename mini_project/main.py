import logging

from game import *
from search.monte_carlo import MonteCarloMixin
from player import *

logging.basicConfig(level=logging.INFO)


class MonteCarloEngine(MonteCarloMixin, InternalEngine):
    pass


if __name__ == '__main__':
    p1 = MonteCarloEngine(colour=chess.WHITE)
    p2 = MonteCarloEngine(colour=chess.BLACK)  #, engine_path='stockfish')

    game = Game(p1, p2)
    game.play()
