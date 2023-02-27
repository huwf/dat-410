import copy
from collections import deque
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

C = 10
SIMULATION_RUNS = 1000


class GameTree:
    def __init__(self, game, rollout_policy=None, score_func=None):
        self.game = game
        self.states = {}
        self.root = State(self.game, last_move=None, parent=None)
        if rollout_policy is None:
            rollout_policy = self.rollout_policy
        self.root.selection_policy = rollout_policy
        self.current_state = copy.deepcopy(self.root)
        if score_func is None:
            score_func = self._score_func
        self.score_func = score_func

    @property
    def states(self):
        """The states which can be played for the _current_ move"""
        return self._states

    @states.setter
    def states(self, value):
        self._states = value

    def rollout_policy(self):
        """Default method for visited nodes

        The only node considered to be "visited" is the root node. All other nodes will likely
        stick to uniform random selection to expand.

        Can be set by setting
        """
        return random.sample(tuple(self.game.board.empty_squares), 1)[0]

    def _score_func(self, state):
        """Default scoring function to evaluate a move

        Takes a State as argument, and evaluates according to the current player in root node
        """
        if state.game.is_draw:
            score = 0.5
        else:
            score = 1 if self.root.game.next_player == state.game.winner else 0
        # Update all states, aside from the first one
        for s in state.traverse():
            s.wins += score

    def simulate(self, state):
        while not state.is_terminal_node:
            pos = state.selection_policy()
            state = state.transition(pos)
        # Backpropagate
        self.score_func(state)


class State:
    """A state in the game - current or possible future"""
    def __init__(self, game, last_move, parent, rollout_policy=None, selection_policy=None):
        self.game = game
        self.last_move = last_move
        self.parent = parent
        self.child_states = {}
        # For unvisited nodes
        if selection_policy is None:
            selection_policy = self._selection_policy
        # For visited (root) nodes:
        if rollout_policy is None:
            rollout_policy = self._selection_policy
        self.rollout_policy = rollout_policy
        self.selection_policy = selection_policy
        self.visits = 0
        self.wins = 0

    @property
    def possible_moves(self):
        return self.game.board.empty_squares

    @property
    def score(self):
        return 0 if not self.visits else self.wins / self.visits

    def transition(self, pos):
        if pos not in self.child_states:
            new_state = State(copy.deepcopy(self.game), last_move=pos, parent=self)
            self.child_states[pos] = new_state
            # Only play the move if the state did not previously exist
            # Otherwise, we only need to transition to it and decide on the move after
            new_state.game.play_turn(new_state.game.next_player, pos)
        new_state = self.child_states[pos]
        new_state.visits += 1

        return new_state
        # self.child_states[pos] = new_state

    def traverse(self):
        ret = deque([])
        ret.appendleft(self)
        s = self
        while (s := s.parent) is not None:
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
        tree = GameTree(game, rollout_policy=self.rollout_policy)
        # Expand the root node to help determine rollout policy
        for pos in game.board.empty_squares:
            state = tree.root.transition(pos)
            tree.simulate(state)

        # Now do the simulation
        for i in range(SIMULATION_RUNS):
            pos = tree.root.rollout_policy()
            state = tree.root.transition(pos)
            logger.debug('[Simulation]')
            tree.simulate(state)
            logger.debug('[End Simulation]')
        return list(tree.root.child_states.values())
        # return tree.root.game.board.empty_squares

    def rollout_policy(self, states):
        scores = {}
        for state in states:
            exploitation = state.wins / state.visits
            exploration = np.sqrt(np.log(state.visits)/state.visits)
            scores[exploitation + C * exploration] = state
        return scores[max(scores.values())]


class RandomSearchMixin:
    def search(self, game):
        return game.board.empty_squares
