import random
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
        return f'Hello. {g}'

    def talk(self, prompt, no_input=False):
        func = print if no_input else input
        return func(prompt)

    def conversation(self):
        state = State(response='')
        say = self.get_greeting()
        while not state.satisfied:
            response = self.talk(say)
            state = StateFactory.get_state(state, response)
            say = state.get_next_prompt()

        self.talk('Finding information, please wait....', no_input=True)
        time.sleep(1)
        self.talk(state.answer(), no_input=True)
        sys.exit(0)

