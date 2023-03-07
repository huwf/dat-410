import random
import re
import sys
import time

from state import State, StateFactory


class Agent:
    def __init__(self, state=None, name=''):
        self.state = state or State('')
        self.history = []
        self.name = name

    def get_greeting(self):
        greetings = [
            'I am a bot.',
            'I am your digital assistant',
            "What do you want?"
        ]
        g = random.randint(0, len(greetings) - 1)

        if self.name:
            return f'Hello, my name is {self.name}. {greetings[g]}'
        return f'Hello. {greetings[g]}'

    def talk(self, prompt, no_input=False):
        func = print if no_input else input
        return func(f'{f"{self.name}: " if self.name else ""}{prompt}\n')

    def yesno(self, response):
        return bool(re.match(r'(yes|correct|y)', response.lower()))

    def conversation(self):
        state = None
        attr = None
        say = self.get_greeting()
        while not state or not state.satisfied:
            try:
                response = self.talk(say)
                state = StateFactory.get_state(state, response)
                state.extract_information(response, attr)
                attr, say = state.get_next_prompt()
            except ValueError as e:
                self.talk('Something has gone wrong, we need to start again', no_input=True)
                self.talk(self.get_greeting())


        confirm = self.talk(f'So just to confirm: {state.confirm_question()}?')
        if not self.yesno(confirm):
            self.talk("Oh dear! I guess we'll have to start again.", no_input=True)
            sys.exit(1)

        self.talk('Finding information, please wait....', no_input=True)
        time.sleep(1)
        self.talk(state.answer(), no_input=True)
        # TODO: Maybe add an extra prompt to start a new conversation
        sys.exit(0)
