import logging

import chess
import chess.engine

logger = logging.getLogger(__name__)


class Player:
    def __init__(self, colour=chess.WHITE, name=None):
        self.colour = colour
        self.name = name or self.colour

    def search(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        pass

    def play(self, game, *args, **kwargs):
        raise NotImplementedError('play not implemented for top class')


class InternalEngine(Player):
    def play(self, game, *args, **kwargs):
        possible_moves = self.search(*args, **kwargs)
        move = self.evaluate(possible_moves, *args, **kwargs)
        game.play(move)


class ExternalEngine(Player):
    def __init__(self, colour=chess.WHITE, engine_path="stockfish"):
        super().__init__(colour=colour)
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    def play(self, game, *_args, **kwargs):
        limit = chess.engine.Limit(time=kwargs.get('time', 1.0))
        result = self.engine.play(game.board, limit)
        return result.move


class HumanPlayer(Player):
    """A class which accepts inputs of a valid SAN move to play.

    Mainly for use in debugging
    """
    def play(self, game, *_args, **_kwargs):
        move = input('Your move:')
        game.board.push_san(move)
        logger.info(move)

