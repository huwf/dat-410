import logging

import chess
import chess.engine

logger = logging.getLogger(__name__)


class Player:
    """A chess player

    Interface with three methods: search, evaluate and play.

    All implementation should be done in a subclass
    """
    def __init__(self, colour=chess.WHITE, name=None):
        self.colour = colour
        self.name = name or ('White' if colour else 'Black')

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    def search(self, game):
        raise NotImplementedError('search not implemented in superclass')

    def evaluate(self, states):
        pass

    def play(self, game, *args, **kwargs):
        raise NotImplementedError('play not implemented for superclass')


class InternalEngine(Player):
    """A generic engine defined in this application

    Only play method is implemented, other methods should be defined in mixins
    """
    def play(self, game, *args, **kwargs):
        possible_moves = self.search(game)
        move = self.evaluate(possible_moves)
        return move


class ExternalEngine(Player):
    """An external chess engine (e.g Stockfish).

    Assumes that it's UCI compatible

    Only play method is implemented, other methods should be defined in mixins
    """
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

