"""The concept of a game of chess, which can be played between two players"""
import logging

import chess
import chess.svg


logger = logging.getLogger(__name__)


class Game:
    """A game of chess

    Thin wrapper around the chess.Board class, with a simple method to manage
    the main state of the game and allow player objects to perform their own
    analysis when it's their turn
    """
    def __init__(self, p1, p2, board=None):
        self.p1 = p1
        self.p2 = p2
        self.board = board or chess.Board()

    def play(self):
        while not self.board.is_game_over():
            # TODO: Don't hardcode this
            player = self.p1 if self.board.turn == self.p1.colour else self.p2
            move = player.play(self)
            logger.info(self.board.san(move))
            self.board.push(move)
            logger.debug(f'\n{self.board}')
        print(self.board.result())
