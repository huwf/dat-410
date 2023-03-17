import csv
import logging
import math
import os

import pandas as pd

from game import *
from mini_project.search.random import RandomInternalEngine
from search.alpha_beta import AlphaBetaMixin

from player import *

from search.monte_carlo import MonteCarloPlayer, MonteCarloModelPlayer

logging.basicConfig(level=logging.INFO)


class MonteCarloEngine(MonteCarloPlayer):
    pass

class AlphaBetaEngine(AlphaBetaMixin, InternalEngine):
    pass


PlayerClass = MonteCarloEngine
RESULTS_PATH = 'results_ucts.100.txt'
EARLY_EXIT = 100

def get_test_set():
    pickle_dir = '../data/pickles'
    test_set = pd.DataFrame()
    for f in os.listdir(pickle_dir):
        if 'out' in f:
            continue
        df = pd.read_pickle(f'{pickle_dir}/{f}')
        df = df.loc[~df['Themes'].str.contains('mateIn1')].iloc[100:500]
        test_set = pd.concat([test_set, df])
        break
    test_set.to_pickle('../data/lichess_test_set.pickle')
    return test_set


def run_eval():
    if not os.path.exists('../data/lichess_test_set.pickle'):
        df = get_test_set()
    else:
        df = pd.read_pickle('../data/lichess_test_set.pickle')
    return df


def get_next_move(p1_cls, p2_cls, row):
    logger.info(f'Starting {row["FEN"]}')
    board = chess.Board(row['FEN'])
    # Our engine is p1, opponent is p2
    p1 = p1_cls(colour=board.turn)  # , stockfish=get_stockfish())
    p2 = p2_cls(colour=(not board.turn))

    game = Game(p1, p2, board)
    game.play(next_move=True)
    move = game.board.pop()
    correct_move = row['Moves'].split()[0]
    return [str(move), correct_move, str(move) == correct_move]


if __name__ == '__main__':
    # p1 = AlphaBetaEngine(colour=chess.WHITE)
    # p2 = AlphaBetaEngine(colour=chess.BLACK)  #, engine_path='stockfish')

    df = run_eval()
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['move_played', 'correct_move', 'is_correct'])
        for i in range(len(df)):
            if EARLY_EXIT and i > EARLY_EXIT:
                print(f'Finishing early, after {i} tests')
                break
            row = df.iloc[i]
            out = get_next_move(PlayerClass, PlayerClass, row)
            writer.writerow(out)
            logger.info(out)
