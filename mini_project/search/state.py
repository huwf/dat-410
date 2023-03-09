import copy
import random
from collections import deque


class BaseState:
    """A state in the game - current or possible future"""

    def __init__(self, game, parent, **_state_kwargs):
        self.game = copy.deepcopy(game)
        self.parent = parent
        self.child_states = {}
        self.visits = 0
        self.wins = 0
        
    @property
    def score(self):
        return 0 if not self.visits else self.wins / self.visits

    def transition(self, pos):
        if pos not in self.child_states:
            new_state = self.State(copy.deepcopy(self.game), parent=self)
            self.child_states[pos] = new_state
            # Only play the move if the state did not previously exist
            # Otherwise, we only need to transition to it and decide on the move after
            new_state.game.play_turn(new_state.game.next_player, pos)
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
        ret = deque([])
        ret.appendleft(self)
        s = self
        while (s := s.parent) is not None:
            ret.appendleft(s)
        return ret




