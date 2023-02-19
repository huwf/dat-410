import pickle

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RuleBasedDiagnosticClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 cell_size_limit=0.95,
                 cell_shape_limit=0.95,
                 cell_texture_limit=0.95,
                 cell_homogeneity_limit=0.95):
        """Rule base diagnostic predictor for breast cancer

        This class provides a simple method for determination of whether a cell
        is benign or malignent through a simple test for "normal" or "abnormal".
        An "abnormal" feature is determined as being above the percentile defined
        in each of the parameters, e.g by default a cell size of higher than the
        95th percentile will be regarded as abnormal.

        :param cell_size_limit:
        :param cell_shape_limit:
        :param cell_texture_limit:
        :param cell_homogeneity_limit:
        """
        self.cell_size_limit = cell_size_limit
        self.cell_shape_limit = cell_shape_limit
        self.cell_texture_limit = cell_texture_limit
        self.cell_homogeneity_limit = cell_homogeneity_limit

    def fit(self, X, y):
        # X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        # Determine what is regarded as normal
        self._area_n_percentile_ = np.percentile(X.area_0, self.cell_size_limit)
        # Standard deviation of the texture (higher variance -> greater chance of malignity)
        self._texture_n_percentile_ = np.percentile(X.texture_1, self.cell_texture_limit)
        # Standard deviation of the smoothness (higher variance -> greater chance of malignity)
        self._smoothness_n_percentile_ = np.percentile(X.smoothness_1, self.cell_shape_limit)
        self._concave_points_n_percentile_ = np.percentile(X['concave points_2'], self.cell_shape_limit)
        self._homogeneity_n_percentile_ = np.percentile(X.symmetry_0, self.cell_homogeneity_limit)

        return self

    def predict(self, X):
        def is_malignant(x):
            return self._is_size_abnormal(x) or \
                self._is_shape_abnormal(x) or \
                self._is_texture_abnormal(x) or \
                self._is_homogeneity_abnormal(x)
        return np.array([is_malignant(X.iloc[i]) for i in range(len(X))])

    def _is_size_abnormal(self, x):
        return x.area_0 >= self._area_n_percentile_

    def _is_shape_abnormal(self, x):
        return x['concave points_2'] > self._concave_points_n_percentile_ or \
            x.smoothness_1 > self._smoothness_n_percentile_

    def _is_texture_abnormal(self, x):
        return x.texture_2 > self._texture_n_percentile_

    def _is_homogeneity_abnormal(self, x):
        return x.symmetry_0 > self._homogeneity_n_percentile_


def setup_data():
    with open('data/wdbc.pkl', 'rb') as f:
        df = pickle.load(f)
        print(df)

        y = df.malignant
        df = df.drop(columns=['malignant'])
        return train_test_split(df, y)

if __name__ == '__main__':
    clf = RuleBasedDiagnosticClassifier()
    X_train, X_test, y_train, y_test = setup_data()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

