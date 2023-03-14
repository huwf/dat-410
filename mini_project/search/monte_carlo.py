"""Module for implementing chess with a Monte-Carlo search tree"""

import logging
import math
import random

from mini_project.evaluate import stockfish
from mini_project.search.state import BaseState
from mini_project.search.tree import GameTree

SIMULATION_RUNS = 10

logger = logging.getLogger(__name__)


class MonteCarloMixin:
    """Mixin to perform a search using MCST

    Assumes that the root node is unexpanded, and so will go through each move
    at least once before adopting a selection policy (defined in MonteCarloState)
    """
    def search(self, game):
        """Perform simulation on SIMULATION_RUNS worth of moves

        By default the rollout/selection policies are random, but this can be
        overwritten to something more sensible
        """

        # Expand the root node to help determine rollout policy
        tree = self._get_eval_distribution(game)

        # Now do the simulation
        for i in range(SIMULATION_RUNS):
            move = tree.root.rollout_policy(tree.root.child_states, i)
            state = tree.root.transition(move)
            logger.debug('[Simulation]')
            tree.simulate(state, depth=math.inf)
            logger.debug('[End Simulation]')

        return list(tree.root.child_states.values())

    def _get_eval_distribution(self, game):
        tree = MonteCarloGameTree(game)
        return self._rollout_distribution(game, tree)

    def _rollout_distribution(self, game, tree):
        for move in game.board.legal_moves:
            state = tree.root.transition(move)
            tree.simulate(state)
        return tree

    def evaluate(self, states):
        """Return the move for the highest score

        :states: A list of MonteCarloStates
        """
        ordered = sorted(states, key=lambda x: x.score, reverse=True)
        return ordered[0].game.board.pop()


class MonteCarloModelMixin(MonteCarloMixin):
    """Get the rollout distribution from a model"""
    def _rollout_distribution(self, game, tree):


        for move in game.board.legal_moves:
            state = tree.root.transition(move)
            tree.simulate(state)
        return tree


class MonteCarloState(BaseState):
    """A game state with MCTS specific functions added"""
    def __init__(self, game, parent):
        super().__init__(game, parent)

    """
    Although setting correct rollout and selection policies is still to be done
    there may be cases where we wish to try different policies. This  should be
    overridden in a subclass, and added to the class instead
    """
    def selection_policy(self, moves):
        """Default method for which square to select

        Selects a random possible move to explore. This does not select the final move that the player
        will play, only the move to simulate in the current state
        """
        return random.sample(tuple(moves), 1)[0]

    def rollout_policy(self, states, _N):
        return random.choice(tuple(states))

    def select(self, legal_moves):
        try:
            if not self.parent:
                return self.rollout_policy(self.game.board.legal_moves, 1)
            return self.selection_policy(self.game.board.legal_moves)
        except AssertionError as e:
            print(str(e))
            raise

    def update_score(self, score):
        self.wins += score
        self.visits += 1


class MonteCarloGameTree(GameTree):
    def score_func(self, game, turn):
        """Play to the end, and see what the result is"""



class MonteCarloStockfishGameTree(MonteCarloGameTree):
    State = MonteCarloState

    def __init__(self, game, rollout_policy=None, selection_policy=None):
        self.rollout_policy = rollout_policy
        self.selection_policy = selection_policy
        # Unfortunately we need to pass selection and rollout policy as kwargs
        # so that MonteCarloState is initialised with them
        super().__init__(game)  #, rollout_policy=rollout_policy, selection_policy=selection_policy)

    def score_func(self, game, turn):
        return stockfish(game.board, turn)

