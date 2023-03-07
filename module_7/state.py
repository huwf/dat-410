import copy
import re

from backend import Weather, Translation, Transport


class StateFactory:
    @staticmethod
    def get_state(prev=None, response=''):
        if prev:
            cls = prev.__class__
        else:
            cls = UnknownState.find_state(response)
        return cls(response, prev)


class State:
    REQUIRED_INFORMATION = {'introduction': True}
    RULES = {}
    PROMPTS = {}

    def __init__(self, response, prev=None):
        if prev:
            self.known_information = copy.deepcopy(prev.known_information) if prev else {}
            self.previous_states = prev.previous_states
            self.previous_states.append(prev)
        else:
            self.known_information = {}
            self.previous_states = []

        self.update_attrs()
        if response:
            self.extract_information(response)

    def extract_information(self, response, attr=None):
        if attr:
            self.known_information[attr] = response
        else:
            for rule, attr in self.RULES.items():
                s = re.search(rule, response)
                if not self.known_information.get(attr):
                    self.known_information[attr] = s if s is None else s[0]
        self.update_attrs()

    def update_attrs(self):
        for k, v in self.known_information.items():
            self.__setattr__(k, v)

    def get_next_prompt(self):
        for k in self.REQUIRED_INFORMATION:
            if self.known_information.get(k) is None:
                return k, self.PROMPTS[k]
        # Probably won't use this prompt, but this keeps the code self-documenting
        return None, 'I have everything I need to know!'

    @property
    def satisfied(self):
        for k in self.REQUIRED_INFORMATION:
            if self.known_information.get(k) is None:
                return False
        return True

    def answer(self):
        return self.backend.answer(**self.known_information)

    def confirm_question(self):
        raise NotImplementedError('confirm_question should be implemented in a subclass')


class WeatherState(State):
    REQUIRED_INFORMATION = {
        'time': None,
        'location': None,
    }

    RULES = {
        r'(?<=in )\w+': 'location',
        r'(?<=for )\w+': 'time'
    }

    PROMPTS = {
        'location': "where do you want to check the weather?",
        'time': "When do you want to know the time for this weather?"
    }

    backend = Weather

    def confirm_question(self):
        return "You want to know what the weather is in " + self.location, "for " + self.time


class TransportState(State):
    REQUIRED_INFORMATION = {
        'from_location': None,
        'to_location': None,
        'time': None,
    }

    Rule = {
        r'(?<=from )\w+': 'from_location',
        r'(?<=to )\w+': 'to_location',
        r'(?<=at )\w+': 'time'
    }

    PROMPTS = {
        'from_location': "Where do you want to go from?",
        'to_location': "Where do you want to go to?",
        'time': "When do you want to arrive?"
    }

    backend = Transport

    def confirm_question(self):
        return f"You want to know the best transport from {self.from_location} to {self.to_location}, " \
               f"arriving before {self.time}?"

class TranslateState(State):
    REQUIRED_INFORMATION = {
        'word': None,
        'from_language': None,
        'to_language': None,
    }

    RULES = {
        r'(?<=from )\w+': 'from_language',
        r'(?<=into )\w+ | (?<=to )\w+': 'to_language',
        r'(?<=what is )\w+ | (?<=translate )\w+': 'word'
    }

    PROMPTS = {
        'from_language': "What language do you want to translate it from?",
        'to_language': "what language do you want to translate it to:",
        'word': "what word do you want to translate: "
    }

    def confirm_question(self):
        return f"You want to translate {self.word} from {self.from_language}into {self.to_language}"

    backend = Translation


class UnknownState(State):
    RULES = {
        r"^.*translate.*$": TranslateState,
        r"^.*weather.*$|^.*temperature.*$": WeatherState,
        # r"^.*transport.*$|^.*bus.*$|^.*tram.*$": TransportState,
        r"^.*transport.*$|^.*get to.*$|^.*tram*$|^.*buss*": TransportState
    }

    @staticmethod
    def find_state(response):
        for rule, cls in UnknownState.RULES.items():
            if re.match(rule, response):
                return cls
        raise ValueError("I'm sorry, I do not know what you mean!")
