import copy
import random
from collections import deque

from mini_project.game import Game


class BaseState:
    """A state in the game - current or possible future

    Contains some common behaviour, but expects subclasses to implement details
    """

    def __init__(self, game, parent, **_state_kwargs):
        # self.game = copy.deepcopy(game)
        self.game = Game(game.p1, game.p2, copy.deepcopy(game.board))
        self.parent = parent
        self.child_states = {}
        self.visits = 0
        self.wins = 0
        
    @property
    def score(self):
        return 0 if not self.visits else self.wins / self.visits

    def transition(self, pos):
        if pos not in self.child_states:
            new_state = self.__class__(copy.deepcopy(self.game), parent=self)
            self.child_states[pos] = new_state
            # Only play the move if the state did not previously exist
            # Otherwise, we only need to transition to it and decide on the move after
            new_state.game.board.push(pos)
        new_state = self.child_states[pos]
        new_state.visits += 1

        return new_state

    def select(self, legal_moves):
        """Interface for selecting a move

        E.g in Monte-Carlo search, this will select according to self.rollout_policy
        if it is a root node, and the self.selection_policy if not
        """
        raise NotImplementedError('select not implemented in superclass')

    def traverse(self):
        """Traverse from the current node up to the root

        A generator which yields each state, starting with the current one
        """
        s = self
        yield self
        while (s := s.parent) is not None:
            yield s

    def update_score(self, score):
        """Update the score for the state

        Assuming a score for the end of a simulation, update the score in a
        custom way. For example:
          self.wins += 1
          self.visits += 1
        """
        raise NotImplementedError('Implement update_score in subclass')
