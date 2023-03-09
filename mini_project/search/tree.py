from mini_project.game import Game
from mini_project.search.state import *


class GameTree:
    """A generic search tree for simulating different moves"""
    State = BaseState

    def __init__(self, game: Game, **state_kwargs):

        self.game = game
        self.root = self.State(game, None, **state_kwargs)
        self.child_states = {}

    def simulate(self, state, depth=10, end_condition_func=None):
        """Simulate for n moves

        :param state: The state to simulate from
        :param depth: How many moves to simulate

        :param end_condition_func: An optional function which can define the
        end condition. This is an `or` statement, so if you wish to set this,
        make sure to set a high `depth`
        """

        end_condition_func = end_condition_func if end_condition_func else lambda: True
        i = 0
        while not state.game.board.is_game_over():
            if i < depth or end_condition_func():
                break
            move = state.select(state.game.board.legal_moves)
            state = state.transition(move)
            i += 1

    def transition(self, move):
        if move not in self.child_states:
            new_state = self.State(self.game, parent=self)
            self.child_states[move] = new_state
            # Only play the move if the state did not previously exist
            # Otherwise, we only need to transition to it and decide on the move after
            new_state.game.board.push_move(move)
        new_state = self.child_states[move]
        new_state.visits += 1

        return new_state
