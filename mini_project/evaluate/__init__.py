"""
This module contains scoring functions, and other evaluation methods

For MCTS the way to evaluate is to use Stockfish to evaluate after a certain
amount of moves and get the win/draw/loss scores from
"""

import chess
from stockfish import Stockfish

stockfish_inst = None
def get_stockfish():
    global stockfish_inst
    if not stockfish_inst:
        stockfish_inst = Stockfish("stockfish")
    return stockfish_inst

# from mini_project.main import stockfish_inst

def stockfish(game, start_player: chess.Color):  # , stockfish_inst=None):
    """Evaluate the position with stockfish

    Assumes the player we want to score is game.p1
    """
    board = game.board
    # s = stockfish_inst or Stockfish(path="stockfish")
    s = get_stockfish()
    fen = board.fen()
    outcome = board.outcome()
    if outcome:
        if outcome.winner == game.p1.colour:
            return 1
        if outcome.winner == game.p2.colour:
            return 0
        return 0.5

    s.set_fen_position(fen)
    wdl = s.get_wdl_stats()
    # print(wdl, s.get_evaluation())
    # print(s.get_top_moves(len(list(board.legal_moves))))
    # It won't necessarily be the same player's move now as it was from the
    # start of the simulation, so we need to get the correct value
    try:
        wins = wdl[0] if (fen.split()[1] == 'w') == start_player else wdl[2]
        return (wins + (0.5 * wdl[1])) / sum(wdl)
    except TypeError as e:
        print(str(e))
        raise



def play_to_end(board, start_player: chess.Color):
    pass



def stockfish_evaluate_all(board):  # , stockfish_inst=None):
    # s = stockfish_inst or Stockfish(path="stockfish")
    s = get_stockfish()
    fen = board.fen()
    s.set_fen_position(fen)
    wdl = s.get_wdl_stats()
    # print(wdl, s.get_evaluation())
    return s.get_top_moves(len(list(board.legal_moves)))
    # It won't necessarily be the same player's move now as it was from the
    # start of the simulation, so we need to get the correct value

