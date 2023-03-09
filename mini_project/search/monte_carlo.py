import random

from mini_project.search.state import BaseState


class MonteCarloState(BaseState):
    def __init__(self, game, parent, rollout_policy=None, selection_policy=None):
        super().__init__(game, parent)
        if parent:
            if not selection_policy:
                selection_policy = parent.selection_policy
            if not rollout_policy:
                rollout_policy = parent.rollout_policy

        self.selection_policy = selection_policy or self._selection_policy
        self.rollout_policy = rollout_policy or self.selection_policy

    def _selection_policy(self):
        """Default method for which square to select

        Selects a random possible move to explore. This does not select the final move that the player
        will play, only the move to simulate in the current state
        """
        return random.sample(tuple(self.game.board.empty_squares), 1)[0]

    def select(self, legal_moves):
        if not self.parent:
            return self.rollout_policy()
        return self.selection_policy()


class MonteCarloGameTree:
    State = MonteCarloState

    def __init__(self, game, rollout_policy=None, selection_policy=None):
        self.rollout_policy = rollout_policy
        self.selection_policy = selection_policy
        # Unfortunately we need to pass selection and rollout policy as kwargs
        # so that MonteCarloState is initialised with them
        super().__init__(game, rollout_policy=rollout_policy, selection_policy=selection_policy)


