import copy
import os
import pickle
import re
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

TRANSLATION_FILES = [
    (('data/europarl-v7.de-en.lc.de', 'German'), ('data/europarl-v7.de-en.lc.en', 'English')),
    # (('data/europarl-v7.fr-en.lc.fr', 'French'), ('data/europarl-v7.fr-en.lc.en', 'English')),
    # (('data/europarl-v7.sv-en.lc.sv', 'Swedish'), ('data/europarl-v7.sv-en.lc.en', 'English'))
]

START_CHAR = '<start>'
NULL_CHAR = 'NULL'


# TODO: We should probably HTMLEncode the source files as &apos; becomes & apos ;


class LanguageModel:
    def __init__(self, language, source):
        self.language = language
        self.source = source
        self.word_counts = None
        self.bigrams = None

    def get_word_counts(self, include_start_char=True):
        counter = Counter()
        for line in self.source_lines:
            # self.source_lines.append(line)
            if include_start_char:
                line = f'{START_CHAR} {line}'
            for word in line.rstrip('\n').split(' '):  # If we want punctuation
                counter[word] += 1
        return counter

    @property
    def source_lines(self):
        with open(self.source, encoding='utf-8') as f:
            for line in f.readlines():
                yield line

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
        for line in self.source_lines:
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
    def __init__(self, source, target, num_iter=10, early_exit=None):
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
        self.model = None
        self.decoded = None
        self.early_exit = early_exit

    def _get_empty_softcount_objs(self):
        c_t_orig = {w: 0 for w in self.target.word_counts}
        c_t_orig[NULL_CHAR] = 0
        c_s_t_orig = {w: {} for w in c_t_orig}
        return c_t_orig, c_s_t_orig

    def _get_random_initialisation(self):
        t_t_s = {}
        for en, (s, t) in enumerate(zip(self.source.source_lines, self.target.source_lines)):  # Each pair
            if isinstance(self.early_exit, int) and en >= self.early_exit:
                break
            t = t.rstrip('\n').split(' ')
            s = s.rstrip('\n').split(' ')
            # TODO: NULL characters can come later
            # if len(s) != len(t):
            #     continue
            s = [NULL_CHAR] + s
            t = [NULL_CHAR] + t
            for i in t:  # For each word in target language
                for j in s:  # For each word in the source language
                    # Initialise randomly on the first run
                    t_t_s = self._assign_random_values(t_t_s, j, i)
                    # t_t_s = self._assign_random_values(t_t_s, i, j)
        return t_t_s

    def _predict_probabilities(self, sentences, c_s, c_s_t):
        # counter = -1
        for sent in sentences:  # Each pair
            counter, tup = sent
            s, t = tup

            if isinstance(self.early_exit, int) and counter >= self.early_exit:
                break
            t = t.rstrip('\n').split(' ')
            s = s.rstrip('\n').split(' ')
            # TODO: NULL characters can come later
            # if len(s) != len(t):
            #     continue
            s = [NULL_CHAR for _ in range(len(t) - len(s))] + s
            t = [NULL_CHAR for _ in range(len(s) - len(t))] + t
            for i in t:  # For each word in target language
                for j in s:  # For each word in the source language
                    # Compute alignment probability
                    delta_k_i_j = self.model[i][j] / (np.sum([self.model[word][j] for word in t]))
                    # delta_k_i_j = self.model[j][i] / (np.sum([self.model[word][i] for word in s]))
                    if not c_s_t[i].get(j):
                        c_s_t[i][j] = 0
                    # Update pseudocount
                    c_s_t[i][j] += delta_k_i_j
                    # Update pseudocount
                    c_s[i] += delta_k_i_j
        return c_s_t

    def train(self):
        self.model = self._get_random_initialisation()  # Translation of the target language given the source

        # Do this here so we don't need to generate them each time
        # If the memory cost is too high we can shift these lines into the first loop
        c_t_orig, c_s_t_orig = self._get_empty_softcount_objs()

        for counter in range(self.T):
            sentences = enumerate(zip(self.source.source_lines, self.target.source_lines))
            c_s = copy.copy(c_t_orig)
            c_s_t = copy.deepcopy(c_s_t_orig)
            self.model = self._predict_probabilities(sentences, c_s, c_s_t)
            print(f"Iteration {counter + 1}: {self.predict_word('european')}")
        return self

    def decode(self):
        decoded = {}
        # translations are P(swedish|english)
        # So we can do max(P(english) * P(swedish|english)) to get the translation
        total_source_word_counts = sum(self.source.word_counts.values())
        for word, translations in self.model.items():
            probs = []
            translation_values = sum(translations.values())
            for translation in list(translations.keys()):
                p_word = self.source.word_counts[translation] / total_source_word_counts
                p_translation_word = translations[translation] / translation_values
                probs.append(p_word * p_translation_word)
            if probs:  # Ignore e.g <start>
                best_translation = list(translations.keys())[probs.index(max(probs))]
                decoded[best_translation] = word
        self.decoded = decoded

    def translate(self, sentence):
        words = sentence.rstrip('\n').split(' ')
        # Assume that the sentence is in the [correct] source language
        translated_words = []
        for word in words:
            translated_words.append(self.decoded.get(word, 'NULL'))
        for i in range(len(translated_words) - 1):
            j = i + 1
            word_i = translated_words[i]
            word_j = translated_words[j]
            if word_i == NULL_CHAR or word_j == NULL_CHAR:
                continue
            # First try P(I|J)
            p_i_j = self.target.bigrams[word_j].get(word_i, 0)
            p_j_i = self.target.bigrams[word_i].get(word_j, 0)
            if p_i_j > p_j_i:
                temp = word_j
                translated_words[i] = translated_words[j]
                translated_words[j] = temp
        print(translated_words)

    def predict_word(self, word):
        return sorted([(k, v) for k, v in self.model[word].items()], key=lambda x: x[1], reverse=True)[:20]

    # def predict(self, sentence, given):
    #     # if isinstance(sentence, str):
    #     #     sentence = sentence.rstrip('\n').split(' ')
    #     # if isinstance(given, str):
    #     #     given = given.rstrip('\n').split(' ')
    #     sentences = [(sentence, given)]
    #     sentences = zip([sentence], [given])
    #     t_t_s = {}
    #     for _ in range(self.T):
    #         c_s, c_s_t = self._get_empty_softcount_objs()
    #         t_t_s = self._predict_probabilities(enumerate(sentences, c_s, c_s_t), c_s, c_s_t)
    #     return t_t_s

    def _assign_random_values(self, existing, word, given):
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
    top = words.most_common(20)
    counter = 0
    for i, w in enumerate(top):
        if re.match(r'[a-z]+', w[0]):
            print(w)
            counter += 1
        if counter >= 10:
            break
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
    long_sentence = """Those with faith in the state as an institution were clearly at odds with the more aggressive
    , and arguably more sensible groups , and they lost badly leading to the situation that the future was set so that 
    for any particular set of facts it would always be possible in the great arenas , stadia , and football pitches to 
    state with great and undisputed authority the quote attributed to the great Noam  Chomsky that The country was 
    founded on the principle that the primary role of government is to protect property from the majority , 
    and so it remains .
    """.lower()
    print(f'P(long sentence) = {get_sentence_probability(model.word_counts, bigrams, long_sentence, smoothing=1)}')


def c(num_iter, early_exit, save_pickle, load_pickle):
    """(c) Translation modeling

    We will now estimate the parameters of the translation model P(f|e).
    Self-check: if our goal is to translate from some language into English, why does our conditional probability seem
    to be written backwards? Why don't we estimate P(e|f) instead?

    Write code that implements the estimation algorithm for IBM model 1. Then print, for either Swedish, German, or
    French, the 10 words that the English word european is most likely to be translated into, according to your estimate
    It can be interesting to look at this list of 10 words and see how it changes during the EM iterations.
    """

    for source, target in TRANSLATION_FILES:
        source, source_lang = source
        target, target_lang = target
        print(f'Modelling {source_lang}\n=======================')
        source = LanguageModel(source_lang, source)
        source.get_bigrams()
        english = LanguageModel(target_lang, target)
        english.get_bigrams()
        translation = TranslationModel(source=source, target=english, num_iter=num_iter, early_exit=early_exit)

        if load_pickle:
            with open(f'module4.pickle.{source_lang}', 'rb') as handle:
                translation.model = pickle.load(handle)
        else:
            translation.train()
            if save_pickle:
                with open(f'module4.pickle.{source_lang}', 'wb') as f:
                    pickle.dump(translation.model, f, protocol=pickle.HIGHEST_PROTOCOL)
        translation.predict_word('european')
        translation.decode()


def d():
    """(d) Decoding.

    Define and implement an algorithm to find a translation, given a sentence in the source language. That is, you
    should try to find E* = argmax_E P(E|F)

    In plain words, for a given source-language sentence F, we want to find the English-language sentence E that has the
    highest probability according to the probabilistic model we have discussed. Using machine translation jargon, we
    call this algorithm the "decoder." In practice, you can't solve this problem exactly and you'll have to come up with
    some sort of approximation.

    Exemplify how this algorithm works by showing the result of applying your translation system to a short sentence
    from the source language.

    As mentioned, it is expected that you will need to introduce a number of assumptions to make this at all feasible.
    Please explain all simplifying assumptions that you have made, and the impact you think that they will have on the
    quality of translations. But why is it an algorithmically difficult problem to find the English sentence that has
    the highest probability in our model?
    """
    english = LanguageModel('English', FILES[-1][0])
    english.get_bigrams()
    swedish = LanguageModel('Swedish', FILES[-2][0])
    swedish.get_bigrams()
    translation = TranslationModel(swedish, english)  # , 5, early_exit=1000)
    if os.path.exists('module4.pickle.Swedish'):
        with open(f'module4.pickle.Swedish', 'rb') as handle:
            translation.model = pickle.load(handle)
    else:
        translation.train()
    translation.decode()
    print(translation.translate("herr talman , fru kommissionär ! grattis , lienemann , till ett utmärkt arbete !"))


NUM_ITER = 10
EARLY_EXIT = 10000
LOAD_PICKLE = False
SAVE_PICKLE = False


if __name__ == '__main__':
    a()
    b()
    c(NUM_ITER, EARLY_EXIT, SAVE_PICKLE, LOAD_PICKLE)
    d()
