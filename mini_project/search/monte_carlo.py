"""Module for implementing chess with a Monte-Carlo search tree"""
import copy
import logging
import math
import os
import random

import tensorflow
from tensorflow.keras.models import load_model

import numpy as np
from evaluate import stockfish
from search.state import BaseState
from search.tree import GameTree

from utils import fen_to_bitboard

from mini_project.player import Player, InternalEngine
from mini_project.train.output_features import square_index

model = None

def _get_model():
    cwd = os.getcwd()
    print(cwd)
    global model
    if not model:
        model = load_model('Model/')
    return model


C = math.sqrt(2)
SIMULATION_RUNS = 100

logger = logging.getLogger(__name__)


class MonteCarloPlayer(InternalEngine):
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
        tree = MonteCarloStockfishGameTree(game)
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


class MonteCarloModelPlayer(MonteCarloPlayer):
    """Get the rollout distribution from a model"""

    def _get_eval_distribution(self, game):
        tree = MonteCarloStockfishModelGameTree(game)
        return self._rollout_distribution(game, tree)

    def _rollout_distribution(self, game, tree):
        m = _get_model()
        bitboard = fen_to_bitboard(game.board.fen())
        distribution = m.predict(tensorflow.convert_to_tensor([np.concatenate([bitboard, np.zeros(60)]).reshape((13, 8, 8))]))[0]
        dist_sum = np.sum([distribution[square_index(m.from_square, m.to_square)] for m in game.board.legal_moves])

        legal_move_sum = 0
        scores = {}

        for m in game.board.legal_moves:
            idx = square_index(m.from_square, m.to_square)
            score = distribution[idx] / dist_sum
            legal_move_sum += score
            state = tree.root.transition(m)
            scores[m] = score

        tree.root.rollout_scores = scores
            # state.visits = 1
            # state.wins = score

        print('legal_move_sum ', legal_move_sum)
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

    # def rollout_policy(self, states, _N):
    #     return random.choice(tuple(states))

    def rollout_policy(self, states, N):
        scores = {}
        for pos, state in states.items():
            exploitation = state.wins / state.visits
            exploration = np.sqrt(np.log(N)/state.visits)
            scores[exploitation + (C * exploration)] = pos
        return scores[max(scores.keys())]

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


class MonteCarloModelState(MonteCarloState):

    @property
    def rollout_scores(self):
        return self._rollout_scores

    @rollout_scores.setter
    def rollout_scores(self, value):
        self._rollout_scores = value

    def rollout_policy(self, states, N):
        scores = {}
        for pos, score in self.rollout_scores.items():
            state = states[pos]
            exploration = np.sqrt(np.log(N) / state.visits)
            scores[score + (C * exploration)] = pos
        return scores[max(scores.keys())]

    # def rollout_policy(self, states, N):
    #     scores = {}
    #     for pos, state in states.items():
    #         exploitation = state.wins / state.visits
    #         exploration = np.sqrt(np.log(N)/state.visits)
    #         scores[exploitation + (C * exploration)] = pos
    #     return scores[max(scores.keys())]



class MonteCarloGameTree(GameTree):
    State = MonteCarloState

    def __init__(self, game, rollout_policy=None, selection_policy=None):
        self.rollout_policy = rollout_policy
        self.selection_policy = selection_policy
        # Unfortunately we need to pass selection and rollout policy as kwargs
        # so that MonteCarloState is initialised with them
        super().__init__(game)  #, rollout_policy=rollout_policy, selection_policy=selection_policy)

    def score_func(self, game, turn):
        """Play to the end, and see what the result is"""
        outcome = game.board.outcome()
        winner = outcome.winner
        if winner == game.p1:
            return 1
        if winner == game.p2:
            return 0
        return 0.5


class MonteCarloStockfishGameTree(MonteCarloGameTree):
    def score_func(self, game, turn):
        return stockfish(game, turn)  #, game.p1.stockfish)


class MonteCarloStockfishModelGameTree(MonteCarloStockfishGameTree):
    State = MonteCarloModelState

