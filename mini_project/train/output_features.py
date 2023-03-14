import chess
import numpy as np

# from utils import BITBOARD_PIECE_ORDER

BITBOARD_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
KNIGHT_MOVES = [17, 15, 10, 6, -17, -15, -10, -6]


def get_queen_positions(end_positions):
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


end_positions = {square: {chess.QUEEN: [], chess.KNIGHT: [], chess.PAWN: []} for square in chess.SQUARES}
end_positions = get_queen_positions(end_positions)
end_positions = get_knight_positions(end_positions)
end_positions = get_promotions(end_positions)


def end_positions_to_array(e):
    length = 0
    for square, obj in e.items():
        length += sum([len(v) for _, v in obj.items()])
    return np.zeros(length)


def start_indexes(end_positions):
    indexes = {}
    idx = 0
    for square in chess.SQUARES:
        indexes[square] = idx
        obj = end_positions[square]
        idx += sum([len(o) for o in obj.values()])
    return indexes


end_position_indexes = start_indexes(end_positions)


def square_index(from_square, to_square):
    base = end_positions[from_square]
    if from_square - to_square not in KNIGHT_MOVES:
        ret = base[chess.QUEEN].index(to_square)
    else:
        ret = base[chess.KNIGHT].index(to_square)
    return end_position_indexes[from_square] + ret

