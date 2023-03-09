"""The concept of a game of chess, which can be played between two players"""
import logging

import chess
import chess.svg


logger = logging.getLogger(__name__)


class Game:
    def __init__(self, p1, p2, board=None):
        self.p1 = p1
        self.p2 = p2
        self.board = board or chess.Board()

    def play(self):
        while not self.board.is_game_over():
            player = self.p1 if self.board.turn == chess.WHITE else self.p2
            move = player.play(self)
            self.board.push(move)
            logger.debug(self.board)
            logger.info(self.board.san(move))
