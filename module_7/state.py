import copy
import re

from backend import Weather


class StateFactory:
    @staticmethod
    def get_state(prev=None, response=''):
        if not prev:
            cls = UnknownState
        else:
            cls = UnknownState.find_state(response)
        return cls(response, prev)


class State:
    REQUIRED_INFORMATION = {'introduction': True}
    RULES = []

    def __init__(self, response, prev=None):
        if prev:
            self.known_information = copy.deepcopy(prev.known_information) if prev else {}
            self.previous_states = prev.previous_states
            self.previous_states.append(prev)
        else:
            self.known_information = {}
            self.previous_states = []

        for k, v in self.known_information.items():
            self.__setattr__(k, v)
        if response:
            self.extract_information(response)

    def extract_information(self, response):
        new_info = {}
        for rule in self.RULES:
            groups = re.match(rule, response)
            new_info.update(groups)
        self.known_information.update(new_info)

    def get_next_prompt(self):
        # TODO: Look in self.known_information and self.REQUIRED_INFORMATION and decide what to ask
        # We can probably generalise a "Tell me <<bla>> about <<blabla>>" prompt
        return 'I need more information!'

    @property
    def satisfied(self):
        return sorted(self.known_information.keys()) == sorted(self.REQUIRED_INFORMATION.keys())

    def answer(self):
        return self.backend.answer(**self.REQUIRED_INFORMATION)

class UnknownState(State):
    @staticmethod
    def find_state(response):
        return State

class WeatherState(State):
    REQUIRED_INFORMATION = {
        'time': None,
        'city': None,
    }

    RULES = [
        r''
    ]

    backend = Weather


class TransportState(State):
    pass


class TranslateState(State):
    pass