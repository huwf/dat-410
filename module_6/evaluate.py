import random


class RandomMoveMixin:
    def evaluate(self, possible_moves):
        return random.choice(tuple(possible_moves))


class MonteCarloMoveMixin:

    def evaluate(self, states):
        ordered = sorted(states, key=lambda x: x.score, reverse=True)
        return ordered[0].last_move
        # return random.choice(tuple(possible_moves))

