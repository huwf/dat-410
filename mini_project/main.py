import csv
import logging
import os

import pandas as pd

from game import *
from search.monte_carlo import MonteCarloMixin, MonteCarloStockfishGameTree, MonteCarloStockfishMixin
from player import *



logging.basicConfig(level=logging.INFO)


class MonteCarloEngine(MonteCarloMixin, InternalEngine):
    pass


class MonteCarloStockfishEngine(MonteCarloStockfishMixin, MonteCarloEngine):
    pass

class MonteCarloStockfishEngineInst(MonteCarloStockfishMixin, MonteCarloEngine):
    pass

# class RandomMonteCarloEngine(MonteCarloEngine):
#     pass


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


if __name__ == '__main__':

 #, engine_path='stockfish')

    df = run_eval()
    with open('results_naive.500.txt', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['move_played', 'correct_move', 'is_correct'])
        for i in range(len(df)):
            row = df.iloc[i]
            logger.info(f'Starting {row["FEN"]}')
            board = chess.Board(row['FEN'])
            # Our engine is p1, opponent is p2
            p1 = MonteCarloStockfishEngineInst(colour=board.turn)  # , stockfish=get_stockfish())
            p2 = MonteCarloStockfishEngine(colour=(not board.turn))

            game = Game(p1, p2, board)
            game.play(next_move=True)
            move = game.board.pop()
            correct_move = row['Moves'].split()[0]
            row = [str(move), correct_move, str(move) == correct_move]
            writer.writerow(row)
            logger.info(row)
