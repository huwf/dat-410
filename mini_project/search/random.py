import random

from mini_project.player import InternalEngine


class RandomInternalEngine(InternalEngine):
    def search(self, game):
        possible_moves = list(game.board.legal_moves)
        return random.sample(possible_moves, 1)

    def evaluate(self, moves):
        """Play the first(only) move returned

        N.B. In this class, it returns a Move, not a State
        """
        return moves[0]
