import chess
import numpy as np

BITBOARD_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


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





