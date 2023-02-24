import random

class RandomMoveMixin:
    def evaluate(self, possible_moves):
        return random.choice(tuple(possible_moves))