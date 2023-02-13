import copy
from random import random
from collections import Counter

import numpy as np
from nltk.tokenize import word_tokenize

FILES = [
    ('data/europarl-v7.de-en.lc.de', 'German'),
    ('data/europarl-v7.fr-en.lc.fr', 'French'),
    ('data/europarl-v7.sv-en.lc.sv', 'Swedish'),
    ('data/europarl-v7.sv-en.lc.en', 'English'),
]
START_CHAR = '<start>'

# TODO: We should probably HTMLEncode the source files as &apos; becomes & apos ;


class LanguageModel:
    def __init__(self, language, source):
        self.language = language
        self.source = source
        self.word_counts = None
        self.bigrams = None
        self.source_lines = []

    def get_word_counts(self, source=None, include_start_char=True):
        source = source or self.source
        assert source, 'No language included for word counts'
        counter = Counter()
        with open(source, encoding='utf-8') as f:
            for line in f.readlines():
                self.source_lines.append(line)
                if include_start_char:
                    line = f'{START_CHAR} {line}'
                for word in line.rstrip('\n').split(' '):  # If we want punctuation
                    counter[word] += 1
            return counter

    @property
    def word_counts(self):
        if not self._word_counts:
            self.word_counts = self.get_word_counts(self.source)
        return self._word_counts

    @word_counts.setter
    def word_counts(self, value):
        self._word_counts = value

    def get_bigrams(self):
        bigrams = {w: Counter() for w in self.word_counts}
        with open(self.source, encoding='utf-8') as f:
            for line in f.readlines():
                prev = START_CHAR
                for word in line.rstrip('\n').split(' '):
                    bigrams[prev][word] += 1
                    prev = word
        return bigrams

    @property
    def bigrams(self):
        if not self._bigrams:
            self.bigrams = self.get_bigrams()
        return self._bigrams

    @bigrams.setter
    def bigrams(self, value):
        self._bigrams = value


class TranslationModel:
    def __init__(self, source, target, num_iter=1):
        """A translation model for a source and a target language

        :param source: An instantiated LanguageModel of the source language
        :type source: LanguageModel
        :param target: A instantiated LanguageModel of the target language
        :type target: LanguageModel
        :param num_iter: The amount of times to run EM
        """
        self.source = source
        self.target = target
        self.T = num_iter

    def train(self):
        NULL_CHAR = 'NULL'
        t_t_s = {}  # Translation of the target language given the source

        for counter in range(self.T):
            assert len(self.source.source_lines) == len(self.target.source_lines), \
                'There are different amounts of sentences between source and target'
            c_s = {w: 0 for w in self.source.word_counts}
            c_t_s = c_s.copy()
            for s, t in zip(self.source.source_lines, self.target.source_lines):  # Each pair
                t = t.rstrip('\n').split(' ')
                s = s.rstrip('\n').split(' ')
                # TODO: NULL characters can come later
                if len(s) != len(t):
                    continue
                for i in t:  # For each word in target language
                    # s = [NULL_CHAR] + s
                    for j in s:  # For each word in the source language
                        # Initialise randomly on the first run
                        if counter == 0:
                            t_t_s = self.assign_random_values(t_t_s, i, j)
                        else:
                            # Compute alignment probability
                            delta_k_i_j = t_t_s[i][j] / (np.sum([t_t_s[word][t] for word in s]))
                            if not c_s[j].get(t):
                                c_s[j][t] = 0
                            # Update pseudocount
                            c_t_s[j][t] += delta_k_i_j
                            # Update pseudocount
                            c_s[j] += delta_k_i_j

            # Update probabilities
            # TODO: Get the right values from the different dicts
            t_t_s = c_t_s / c_s
        return t_t_s


    def assign_random_values(self, existing, word, given):
        """Initial assignment for language alignment values"""
        if given not in existing:
            existing[given] = {word: random()}
        if word not in existing[given]:
            existing[given][word] = random()
        return existing



def get_language_frequencies(filename, include_start_char=False):
    counter = Counter()
    with open(filename, encoding='utf-8') as f:
        for line in f.readlines():
            if include_start_char:
                line = f'{START_CHAR} {line}'

            for word in line.rstrip('\n').split(' '):  # If we want punctuation
                counter[word] += 1
        return counter


def get_top_10(words, language):
    print(f'10 most common words for {language}\n--------------------------------')
    top = words.most_common(10)
    for w in top:
        print(w)
    return top


def get_sentence_probability(words, bigrams, sentence, smoothing=1):
    prev = START_CHAR
    p = 0
    # p = 1
    for word in word_tokenize(sentence):
        prev_count = words.get(prev, smoothing)
        p_w_given_prev = (bigrams.get(prev, {}).get(word, smoothing) + smoothing) / (prev_count + (len(words) * smoothing))
        # Add log probability is the same as p *= p_w_given_prev
        p += np.log(p_w_given_prev)
        # p *= p_w_given_prev
        prev = word
    # return p
    return np.exp(p)


def a():
    """
    (a) Warmup. As a warmup, write code to collect statistics about word frequencies in the two languages.
    Print the 10 most frequent words in each language.
    If you're working with Python, using a Counter is probably the easiest solution

    Let's assume that we pick a word completely randomly from the European parliament proceedings. According to your
    estimate, what is the probability that it is speaker? What is the probability that it is zebra?
    """
    counts = {}
    cls = None
    for f, language in FILES:
        cls = LanguageModel(language, f)
        counts = cls.word_counts
        # counts = get_language_frequencies(f)
        get_top_10(cls.word_counts, language)

    # English is the last in FILES
    for word in ["speaker", "zebra"]:
        chance = counts.get(word, 0) / sum(counts.values())
        print(f'Probability of any given word being "{word}": {chance}')


def b():
    """(b) Language modeling.
    We will now define a language model that lets us compute probabilities for individual English sentences.
    Implement a bigram language model as described in the lecture, and use it to compute the probability of a
    short sentence.

    What happens if you try to compute the probability of a sentence that contains a word that did not appear in the
    training texts? And what happens if your sentence is very long (e.g. 100 words or more)? Optionally, change your
    code so that it can handle these challenges.
    """
    # words = get_language_frequencies(FILES[-1][0], include_start_char=True)
    model = LanguageModel('English', FILES[-1][0])
    bigrams = model.get_bigrams()
    # bigrams = get_bigrams(FILES[-1][0], words)
    sentence = "i would like your advice about rule 143 concerning inadmissibility ."
    print(f'P({sentence}) = {get_sentence_probability(model.word_counts, bigrams, sentence, smoothing=1)}')
    long_sentence = """Those that had faith in the state as an institution were clearly at odds with the more aggressive , and
    arguably more sensible groups , and they lost badly leading to the situation that the future was set so that for any 
    particular set of facts it would always be possible in the great arenas , stadia , and football pitches to state 
    with great and undisputed authority the quote attributed to the great Noam  Chomsky that The country was founded on 
    the principle that the primary role of government is to protect property from the majority , and so it remains 
    """.lower()
    print(f'P(long sentence) = {get_sentence_probability(model.word_counts, bigrams, long_sentence, smoothing=1)}')


def c():
    """(c) Translation modeling

    We will now estimate the parameters of the translation model P(f|e).
    Self-check: if our goal is to translate from some language into English, why does our conditional probability seem
    to be written backwards? Why don't we estimate P(e|f) instead?

    Write code that implements the estimation algorithm for IBM model 1. Then print, for either Swedish, German, or
    French, the 10 words that the English word european is most likely to be translated into, according to your estimate.
    It can be interesting to look at this list of 10 words and see how it changes during the EM iterations.
    """
    english = LanguageModel('English', 'data/europarl-v7.sv-en.lc.en')
    english.get_bigrams()
    swedish = LanguageModel('Swedish', 'data/europarl-v7.sv-en.lc.sv')
    swedish.get_bigrams()
    translation = TranslationModel(english, swedish)
    translation.train()
    return translation

if __name__ == '__main__':
    a()
    b()
    out = c()
    print(out)
