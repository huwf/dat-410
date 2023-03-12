import logging

from game import *
from search.monte_carlo import MonteCarloMixin
from search.alpha_beta import AlphaBetaMixin
from player import *

logging.basicConfig(level=logging.INFO)


class MonteCarloEngine(MonteCarloMixin, InternalEngine):
    pass

class AlphaBetaEngine(AlphaBetaMixin, InternalEngine):
    pass


if __name__ == '__main__':
    p1 = AlphaBetaEngine(colour=chess.WHITE)
    p2 = AlphaBetaEngine(colour=chess.BLACK)  #, engine_path='stockfish')

    game = Game(p1, p2)
    game.play()
