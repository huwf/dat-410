import logging

from game import *
from player import *

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    p1 = HumanPlayer(colour=chess.WHITE)
    p2 = ExternalEngine(colour=chess.BLACK, engine_path='stockfish')

    game = Game(p1, p2)
    game.play()
