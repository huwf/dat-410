import logging

from search.state import *


logger = logging.getLogger(__name__)


class GameTree:
    """A generic search tree for simulating different moves"""
    State = BaseState

    def __init__(self, game: Game, **state_kwargs):
        self.game = game
        self.root = self.State(game, None, **state_kwargs)
        self.child_states = {}

    def score_func(self, game, turn):
        raise NotImplementedError('Must implement score func in evaluate.__init__')

    def simulate(self, state, depth=10, end_condition_func=None, backpropagate=True):
        """Simulate for n moves

        :param state: The state to simulate from
        :param depth: How many moves to simulate
        :param end_condition_func: An optional function which can define the
        end condition. This is an `or` statement, so if you wish to set this,
        make sure to set a high `depth`
        :param backpropagate: Whether to backpropagate the results of the
        simulation to each state
        """

        end_condition_func = end_condition_func if end_condition_func else lambda: False
        i = 0
        while not state.game.board.is_game_over():
            if i > depth or end_condition_func():
                break
            try:
                move = state.select(state.game.board.legal_moves)
                logger.debug(move)
            except AssertionError as e:
                print(str(e))
                raise
            state = state.transition(move)
            i += 1
        score = self.score_func(state.game, self.game.board.turn)
        if backpropagate:
            self.backpropagate(state, score)

    def backpropagate(self, state, score):
        """Update score for all states between current and root

        Assumes that the State has got an update_score method which takes a
        single value and applies it to all values in the traverse generator
        """
        for s in state.traverse():
            s.update_score(score)

