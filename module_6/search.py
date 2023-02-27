
class MonteCarloMixin:
    def search(self, board):
        return board.empty_squares
            





class RandomSearchMixin:
    def search(self, board):
        return board.empty_squares