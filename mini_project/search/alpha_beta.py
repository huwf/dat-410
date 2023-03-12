"""Module for implementing alpha beta trees"""

import logging
import random

from evaluate import stockfish
from search.state import BaseState
from search.tree import GameTree

logger = logging.getLogger(__name__)

ALPHA = float("-inf")
BETA = float("inf")

class AlphaBetaMixin:
    """Mixin to perform a search using MCST

    Assumes that the root node is unexpanded, and so will go through each move
    at least once before adopting a selection policy (defined in MonteCarloState)
    """
    def search(self, game):
        """Perform simulation on SIMULATION_RUNS worth of moves

        By default the rollout/selection policies are random, but this can be
        overwritten to something more sensible
        """
        tree = AlphaBetaGameTree(game)
        # Expand the root node to help determine rollout policy
#        for move in game.board.legal_moves:
#            state = tree.root.transition(move)
#            tree.simulate(state)
        
        best_move_score = ALPHA
        best_move = None


        for move in game.board.legal_moves:
            state = tree.root.transition(move)
            self.min_max(state, 10, ALPHA, BETA)
      

        return list(tree.root.child_states.values())
    

    def min_max(self, state, depth, alpha, beta):
        if depth == 0 or state.game.board.is_game_over():
            return state.score
        if state.game.board.turn:
            value = ALPHA
            for move in state.game.board.legal_moves:
                value = max(value, self.min_max(state.transition(move), depth - 1, alpha, beta))
                state.update_score(value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = BETA
            for move in state.game.board.legal_moves:
                value = min(value, self.min_max(state.transition(move), depth - 1, alpha, beta))
                state.update_score(value)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def evaluate(self, states):
        """Return the move for the highest score
        """
        ordered = sorted(states, key=lambda x: x.score, reverse=True)
        return ordered[0].game.board.pop()

class AlphaBetaState(BaseState):
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

class AlphaBetaGameTree(GameTree):
    State = AlphaBetaState

    def __init__(self, game, rollout_policy=None, selection_policy=None):
        self.rollout_policy = rollout_policy
        self.selection_policy = selection_policy
        # Unfortunately we need to pass selection and rollout policy as kwargs
        # so that MonteCarloState is initialised with them
        super().__init__(game)  #, rollout_policy=rollout_policy, selection_policy=selection_policy)

    def score_func(self, game, turn):
        return stockfish(game, turn)
