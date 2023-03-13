"""
This module contains scoring functions, and other evaluation methods

For MCTS the way to evaluate is to use Stockfish to evaluate after a certain
amount of moves and get the win/draw/loss scores from
"""

import chess
from stockfish import Stockfish

from mini_project.game import Game


def stockfish(board, start_player: chess.Color, stockfish_inst=None):
    s = stockfish_inst or Stockfish(path="stockfish")
    fen = board.fen()
    s.set_fen_position(fen)
    wdl = s.get_wdl_stats()
    # print(wdl, s.get_evaluation())
    # print(s.get_top_moves(len(list(board.legal_moves))))
    # It won't necessarily be the same player's move now as it was from the
    # start of the simulation, so we need to get the correct value
    wins = wdl[0] if (fen.split()[1] == 'w') == start_player else wdl[2]
    return (wins + (0.5 * wdl[1])) / sum(wdl)


def stockfish_evaluate_all(board, stockfish_inst=None):
    s = stockfish_inst or Stockfish(path="stockfish")
    fen = board.fen()
    s.set_fen_position(fen)
    wdl = s.get_wdl_stats()
    # print(wdl, s.get_evaluation())
    return s.get_top_moves(len(list(board.legal_moves)))
    # It won't necessarily be the same player's move now as it was from the
    # start of the simulation, so we need to get the correct value

