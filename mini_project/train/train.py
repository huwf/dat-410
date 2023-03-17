import copy
import sys

import chess
import numpy as np
import pandas as pd
from stockfish import Stockfish
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam
from mini_project.evaluate import stockfish, stockfish_evaluate_all
from mini_project.train.output_features import end_positions, square_index, end_positions_to_array


def get_policy_distribution(board, res, output_array):
    """Given a board and a result from Stockfish, calculate a distribution

    As a crude way of calculating a distribution, we take the sum of all centipawn ratings strictly better than drawing
    and calculate the percentage of each individual move
    """
    # if board.turn:  # White to play
    #     # try:
    #     moves = [m for m in res if m['Centipawn'] and m['Centipawn'] > 0]
    #     # except TypeError as e:
    #     #     print(str(e))
    # else:
    #     moves = [np.abs(m['Centipawn']) for m in res if m['Centipawn'] and m['Centipawn'] < 0]
    # try:
    #     move_sum = sum([m['Centipawn'] for m in moves])
    # except IndexError as e:
    #     print(str(e))
    scores = [m['Centipawn'] for m in res if m.get('Centipawn') is not None]
    if not len(scores):
        return output_array
    if board.turn:  # White to play
        low = min(scores)
        scores = np.array(scores) + np.abs(low) + 1
    else:
        high = max(scores)
        scores = np.array(scores) - (np.abs(high) + 1)
    move_sum = np.sum(scores)
    i = 0
    for m in res:
        if m.get('Centipawn') is None:
            continue
        move = chess.Move.from_uci(m['Move'])
        output_array[square_index(move.from_square, move.to_square)] = scores[i] / move_sum
        i += 1
    return output_array


# import numpy as np
# from matplotlib import pyplot as plt
# import tensorflow.keras as keras
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.models import Sequential
#
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# from tensorflow.keras import applications
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input


import tensorflow as tf
print(tf.keras)


def inputs(df):
    return tf.convert_to_tensor([np.concatenate([a, np.zeros(60)]).reshape((13, 8, 8)) for a in df[1]])


def outputs(df):
    try:
        return tf.convert_to_tensor(list(df))
    except ValueError as e:
        print(str(e))
        raise

    # ret = df.to_numpy()
    # ret = tf.convert_to_tensor([list(a) for a in ret])
    # return ret

def make_convnet(X_train, y_train, X_test, y_test, flatten_first=False):
    model = Sequential()
    model.add(Conv2D(10, 3, input_shape=(13, 8, 8), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2928, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(
                learning_rate=0.0001)
            )

    model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=100)
    return model


if __name__ == '__main__':
    from utils import *
    #bugfix_input_output_df(reverse=True)
    #get_input_output_df(reverse=True)

    import os
    print(os.getcwd())
    ins = inputs(
        pd.read_pickle('train/df_in.pickle')
    )
    outs = outputs(
        pd.read_pickle('train/df_out.pickle')
    )
    
    X_train = ins[:8000]
    X_test = ins[800:10000]
    y_train = outs[:8000]
    y_test = outs[8000:10000]
    print(X_train[0].shape)
    net = make_convnet(X_train, y_train, X_test, y_test)
    print(net)



    puzzle = df.iloc[32]
    board = chess.Board(puzzle['FEN'])
    print('White' if board.turn else 'Black')
    print(puzzle['bitboard'])
    s = Stockfish('stockfish')
    for move in board.legal_moves:
        new_board = copy.deepcopy(board)
        new_board.push(move)
        res = stockfish(new_board, board.turn, s)
    res = stockfish_evaluate_all(board, board.turn, s)
    output_array = end_positions_to_array(end_positions)
    get_policy_distribution(board, res, output_array)
