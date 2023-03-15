"""
This module contains scoring functions, and other evaluation methods

For MCTS the way to evaluate is to use Stockfish to evaluate after a certain
amount of moves and get the win/draw/loss scores from
"""

import chess
from stockfish import Stockfish

from game import Game


def stockfish(game: Game, start_player: chess.Color):
    s = Stockfish(path=r"C:\Users\haqvi\stockfish_15.1_win_x64_avx2\stockfish-windows-2022-x86-64-avx2")
    fen = game.board.fen()
    s.set_fen_position(fen)
    wdl = s.get_wdl_stats()
    # It won't necessarily be the same player's move now as it was from the
    # start of the simulation, so we need to get the correct value
    wins = wdl[0] if (fen.split()[1] == 'w') == start_player else wdl[2]
    return (wins + (0.5 * wdl[1])) / sum(wdl)

