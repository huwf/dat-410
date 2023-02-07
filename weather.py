import numpy as np
import pandas as pd

from kmeans import KMeans, KMeansClassifier

USABLE_FEATURES = ['DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws', 'precipitation']


def get_xy(df):
    """Only care about non-categorical features and split the y"""
    return df[USABLE_FEATURES], df['PM_HIGH']


if __name__ == '__main__':

    beijijg_X, beijing_y = get_xy(pd.read_csv('data/Beijing_labeled.csv'))
    shenyang_X, shenyang_y = get_xy(pd.read_csv('data/Shenyang_labeled.csv'))
    X_train = pd.concat([beijijg_X, shenyang_X])
    y_train = pd.concat([beijing_y, shenyang_y])
    # To save running np.fit (k=2), which can take a bit of time, use these precomputed centroids:
    centroids = np.array([
        np.array([-1.15258621e+01,  2.69079310e+01,  1.02154310e+03, 8.31896552e+00,  2.05525172e+02,  1.01724138e-01]),
        np.array([1.64519611e+00,  4.32586146e+01,  1.01632961e+03, 1.57373156e+01,  1.42319467e+01,  4.56998920e-02])
    ])
    clf = KMeansClassifier(k=2, precomputed_centroids=centroids)
    clf.fit(X_train, y_train)
    print(f'y_train: {clf.score(X_train, y_train)}')

    guangzhou_X, guangzhou_y = get_xy(pd.read_csv('data/Guangzhou_labeled.csv'))
    shanghai_X, shanghai_y = get_xy(pd.read_csv('data/Shanghai_labeled.csv'))
    X_test = pd.concat([guangzhou_X, shanghai_X])
    y_test = pd.concat([guangzhou_y, shanghai_y])

    print(f'y_test: {clf.score(X_test, y_test)}')