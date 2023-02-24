
class MonteCarloMixin:
    pass


class RandomSearchMixin:
    def search(self, board):
        return board.empty_squares