import csv
import os

import chess
import numpy as np
import pandas as pd
from stockfish import Stockfish

from mini_project.evaluate import stockfish_evaluate_all
from mini_project.train.train import get_policy_distribution
from mini_project.utils import fen_to_bitboard

COLUMNS = [
    'PuzzleId',
    'FEN',
    'Moves',
    'Rating',
    'RatingDeviation',
    'Popularity',
    'NbPlays',
    'Themes',
    'GameUrl',
    'OpeningFamily',
    'OpeningVariation'
]
SUBSET = 100

def preprocess_csv(path):
    """Pandas doesn't like rows of different lengths for some reason, so add an
     empty column to rows with only 10 columns
     """
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        with open(f'{path}.processed', 'w', encoding='utf-8') as f2:
            writer = csv.writer(f2)
            writer.writerow(COLUMNS)
            for row in reader:
                if len(row) == 10:
                    row.append('')
                writer.writerow(row)


def get_last_move_from_bitboard_uci(before, after):
    a = before[:64*6].astype(bool)
    b = after[:64*6].astype(bool)
    xor = np.logical_xor(a, b)
    which = np.where(xor == True)[0]
    # TODO: Probably doesn't apply for castling
    assert len(which) == 2, 'There should only be two squares changed'
    square_numbers = which - ((which // 64) * 64)
    # If the before array has a piece on the first value we know the piece started there
    # Otherwise, it will have started somewhere else
    move_arr = square_numbers if before[which[0]] else np.flip(square_numbers)
    return "".join([chess.square_name(s) for s in move_arr])


def get_puzzle(df):
    """Construct a puzzle from the dataframe

    The puzzles are in a format where they are one move before the puzzle start
    So we need to obtain the puzzle, make a move and then return it
    """
    fen = df['FEN']
    moves = df['Moves'].split()
    board = chess.Board(fen)
    board.push_uci(moves[0])
    df['FEN'] = board.fen()
    df['Moves'] = " ".join(moves[1:])
    # Push to get the optimal move
    df['bitboard'] = fen_to_bitboard(board.fen()).astype('object')
    board.push_uci(moves[1])
    new_fen = board.fen()
    df['output'] = fen_to_bitboard(new_fen, reverse_order=True).astype('object')
    df['winning_fen'] = new_fen
    return df


def write_pickle():
    pickle_dir = '../../data/pickles'
    with pd.read_csv('../../data/lichess_db_puzzle.csv.processed', header=0, names=COLUMNS, chunksize=10000) as reader:
        for i, df in enumerate(reader):
            pickle_file = f'{pickle_dir}/lichess_db_puzzle.csv.processed.{i}.pickle'
            if os.path.exists(pickle_file):
                continue
            df = df.apply(get_puzzle, axis=1)
            df.to_pickle(pickle_file)
            df = None


def get_input_output_df(reverse=False):
    s = Stockfish('stockfish')
    from mini_project.train.output_features import end_positions, end_positions_to_array
    from mini_project.train.train import get_policy_distribution
    pickle_dir = '../../data/pickles'
    new_pickles = '../../data/new_pickles'
    it = os.listdir(pickle_dir)
    for f in it if not reverse else np.flip(it):
        print(f)
        if os.path.exists(f'{new_pickles}/100_in_{f}'):
            print('Already exists')
            continue
        df = pd.read_pickle(f'{pickle_dir}/{f}')
        # Exclude mate in 1 because they have many solutions
        # df = df.loc[~df['Themes'].str.contains('mateIn1')]
        # TODO: Temporary, only get first 1000 so that we can get something finished at least
        df = df.loc[~df['Themes'].str.contains('mateIn1')].iloc[:SUBSET]
        arr = df[['PuzzleId', 'bitboard', 'FEN']].to_numpy()
        output_arrays = np.array([end_positions_to_array(end_positions) for _ in range(len(arr))]).astype('object')
        for i in range(len(arr)):
            fen = arr[0:, 2][i]
            board = chess.Board(fen)
            res = stockfish_evaluate_all(board)
            output_arrays[i] = get_policy_distribution(board, res, output_arrays[i])
        pd.DataFrame(arr).to_pickle(f'{new_pickles}/{SUBSET}_in_{f}')
        pd.DataFrame(output_arrays).to_pickle(f'{new_pickles}/{SUBSET}_out_{f}')


def bugfix_input_output_df(reverse=False):
    pickle_dir = '../../data/new_pickles'
    concat_in = pd.DataFrame()
    concat_out = pd.Series()
    it = os.listdir(pickle_dir)
    for f in it if not reverse else np.flip(it):
        if 'out' in f:
            continue
        print(f)
        f_in = f'{pickle_dir}/{f}'
        df_in = pd.read_pickle(f_in)
        df_out = pd.read_pickle(f_in.replace('in', 'out')).to_numpy()
        new_arr_out = []
        indexes = []
        for i in range(len(df_out)):
            try:
                _ = np.where(df_out[i] > 0)
                new_arr_out.append(df_out[i].astype('float32'))
                indexes.append(i)
            except ValueError:
                print(f'Fixing {i}')
                board = chess.Board(df_in.iloc[i][2])
                res = stockfish_evaluate_all(board)
                output_array = df_out[i]
                output_array = get_policy_distribution(board, res, output_array)
                new_arr_out.append(output_array.astype('float32'))
        concat_in = pd.concat([concat_in, df_in[df_in.index.isin(indexes)]])
        concat_out = pd.concat([concat_out, pd.Series(new_arr_out)])
    concat_in.to_pickle(f'df_in.pickle{".reverse" if reverse else ""}')
    concat_out.to_pickle(f'df_out.pickle{".reverse" if reverse else ""}')



if __name__ == '__main__':
    bugfix_input_output_df()
    get_input_output_df()


    # path = '../../data/lichess_db_puzzle.csv'
    # to_bitboard()
    # preprocess_csv(path)

    # path = '../../data/lichess_db_puzzle.csv.processed'
    # i = 0
    # full_pickle = '../../data/lichess_db_puzzle_correct_move.pickle'
    # if not os.path.exists('../../data/lichess_db_puzzle_correct_move.pickle'):
    #     write_pickle()
    #     df = pd.concat([pd.read_pickle(f'../../data/pickles/{f}') for f in os.listdir('../../data/pickles')])
    #     df.to_pickle(full_pickle)
    # # df = df.apply(get_puzzle, axis=1)
    # # df.to_pickle('../../data/lichess_db_puzzle_correct_move.pickle')
    #
    # df = pd.read_pickle(full_pickle)
    # subset = df.head(100)
    #
    # subset = subset.apply(get_puzzle, axis=1)
    # subset.to_pickle('../../data/lichess_db_puzzle_head.pickle')
    # # subset.to_csv('../../data/lichess_db_puzzle_head.csv')
    # subset = pd.read_pickle('../../data/lichess_db_puzzle_head.pickle')
    # bitboard = subset.iloc[0]['bitboard'][0]
    # output = subset.iloc[0]['output'][0]
    # print(subset)
