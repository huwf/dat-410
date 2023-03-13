import csv
import os

import chess
import numpy as np
import pandas as pd
from stockfish import Stockfish

from mini_project.evaluate import stockfish, stockfish_evaluate_all

BITBOARD_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
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
KNIGHT_MOVES = [17, 15, 10, 6, -17, -15, -10, -6]
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


def fen_to_bitboard(fen, reverse_order=False):
    """Takes a FEN position, and returns an array representing the board
    :param fen: A string in FEN format indicating the current state of play
    See https://en.wikipedia.org/wiki/Forsyth-Edwards_Notation
    :return: array of length 772 in the following format:
    For each piece, blocks of 64 for an 8 x 8 chess board where 1 indicates
    the presence of the piece, and 0 indicates the absence.
    The first 6 * 64 in the array is the pieces for the player to move, the
    second 6 * 64 is the opponent, the final four squares indicate whether
    castling is allowed (current player first) for king's side and queen's side
    Piece order: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
    """
    board = chess.Board(fen)
    # 64 squares, 6 types of piece, 2 players, 2 types of castling (x2)
    arr_size = (64 * 6 * 2) + 4
    arr = np.zeros(arr_size, dtype='int')
    # So we know whose turn it is, we put the next player's pieces first
    order = [chess.WHITE, chess.BLACK] if board.turn == chess.WHITE else [chess.BLACK, chess.WHITE]
    if reverse_order:
        order = [not c for c in order]
    for i, colour in enumerate(order):

        for j, piece in enumerate(BITBOARD_PIECE_ORDER):
            for k in list(board.pieces(piece, colour)):
                idx = (i * arr_size // 2) + (j * 64) + k
                arr[idx] = 1
        castle_idx = arr_size - 4 + (2 * i)
        arr[castle_idx] = board.has_kingside_castling_rights(colour)
        arr[castle_idx + 1] = board.has_queenside_castling_rights(colour)
    return arr


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


def get_queen_positions(end_positions):
    print('get_queen_positions')
    for square in chess.SQUARES:
        board = chess.Board()
        board.clear_board()
        board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
        idx = chess.QUEEN
        end_positions[square][idx] = [m.to_square for m in board.legal_moves]
    return end_positions


def get_knight_positions(end_positions):
    # If a knight starts from the middle, it can go these squares (assuming 0-63)
    # Some will be illegal (e.g if they are negative or on the edge of the board)
    # But these will always have a probability of 0
    for square in chess.SQUARES:
        for i in KNIGHT_MOVES:
            end_positions[square][chess.KNIGHT].append(square + i)
    return end_positions


def get_promotions(end_positions):
    for square in chess.SQUARES:
        # print(square)
        # Capture left, advance, capture right (even if illegal)
        for s in [7, 8, 9]:
            for _ in BITBOARD_PIECE_ORDER[1:]:  # Could get all of these possible pieces
                end_positions[square][chess.PAWN].append(square + s)
    return end_positions





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


def get_input_output_df():
    s = Stockfish('stockfish')
    from mini_project.train.output_features import end_positions, end_positions_to_array
    from mini_project.train.train import get_policy_distribution
    pickle_dir = '../../data/pickles'
    new_pickles = '../../data/new_pickles'

    for f in os.listdir(pickle_dir):
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
            res = stockfish_evaluate_all(board, stockfish_inst=s)
            output_arrays[i] = get_policy_distribution(board, res, output_arrays[i])
        pd.DataFrame(arr).to_pickle(f'{new_pickles}/{SUBSET}_in_{f}')
        pd.DataFrame(output_arrays).to_pickle(f'{new_pickles}/{SUBSET}_out_{f}')


if __name__ == '__main__':
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
