import copy
from collections import deque
import random

class GameTree:
    def __init__(self, game, selection_policy=None):
        self.game = game
        self.states = {}
        self.root = State(self.game, None)
        self.current_state = copy.deepcopy(self.root)
        self.selection_policy = selection_policy

    @property
    def states(self):
        """The states which can be played for the _current_ move"""
        return self._states

    @states.setter
    def states(self, value):
        self._states = value

    def simulate(self, state):
        while not state.is_terminal_node:
            pos = state.selection_policy()
            state = state.transition(pos)


class State:
    """A state in the game - current or possible future"""
    def __init__(self, game, parent, selection_policy=None):
        self.game = game
        self.parent = parent
        self.child_states = {}
        if selection_policy is None:
            selection_policy = self._selection_policy
        self.selection_policy = selection_policy

    @property
    def possible_moves(self):
        return self.game.board.empty_squares

    def transition(self, pos):
        if pos not in self.child_states:
            self.child_states[pos] = State(copy.deepcopy(self.game), self)
        new_state = self.child_states[pos]
        new_state.game.play_turn(self.game.next_player, pos)
        return new_state
        # self.child_states[pos] = new_state

    def traverse(self):
        ret = deque([])
        ret.appendleft(self)
        s = self
        while s := s.parent is not None:
            ret.appendleft(s)
        return ret

    @property
    def is_terminal_node(self):
        return self.game.is_finished or not self.game.board.empty_squares

    def _selection_policy(self):
        """Default method for which square to select

        Selects a random possible move to explore. This does not select the final move that the player
        will play, only the move to simulate in the current state
        """
        return random.sample(tuple(self.game.board.empty_squares), 1)[0]



class MonteCarloMixin:
    def search(self, game):
        tree = GameTree(game)
        for i in range(5):
            pos = tree.root.selection_policy()
            state = tree.root.transition(pos)
            tree.simulate(state)
        return tree .root.game.board.empty_squares

    def rollout_policy(self):
        return 0

class RandomSearchMixin:
    def search(self, game):
        return game.board.empty_squares