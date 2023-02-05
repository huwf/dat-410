import math
from collections import Counter
from math import inf

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state


class KMeans(BaseEstimator):
    def __init__(
            self,
            k,
            init='random',
            n_init=10,
            distance='euclidean',
            precomputed_centroids=None,
            max_iter=50,
            precision=0.0001,
            quality='sse',
            random_state=None
    ):
        """Estimator for k-means clustering

        :param k: Amount of clusters to fit data to
        :param init: Method to use for initialising the centroids. Can be the
        string "random", or a function to use calculate. Ignored if
         `precomputed_centroids` is not None
        :param n_init: An integer specifying how many times the initialisation
         should run to determine the best cluster
        :param distance: A string or a function which defines the measure to
         decide which centroid is closest. Defaults to "euclidean", which is
         scipy.spatial.distance.euclidean, but can also be any function
         accepting two array-like parameters
        :param precomputed_centroids: (optional) k n-dimensional coordinates
         which can be used as centroids for the fit without having to calculate
        :param max_iter: The amount of iterations to run the k-means algorithm
         on the data after initialisation. Defaults to 200
        :param precision: The level of precision to regard two points as being
         equal. Defaults to 0.0001
        :param quality: Function to be used to calculate the score. Will
         default to in-cluster SSE, which is summed for the overall score.
        :param random_state: (int or RandomState) If set, will use this value
         as a seed using the numpy RandomState class
        """
        # Define self.X properly later in self.fit
        self.X = np.array([])
        self.k = k
        self.init = init
        self._init_method = \
            self._compute_random_centroids if init == 'random' else self.init
        self.n_init = n_init
        if distance == 'euclidean':
            distance = sp.spatial.distance.euclidean
        self.distance = distance
        self.precomputed_centroids = precomputed_centroids
        self.centroids = self.precomputed_centroids
        self.max_iter = max_iter
        self.precision = precision
        if random_state is None:
            self.random_state = None
        else:
            self.random_state = check_random_state(random_state)
        if quality is None or quality == 'sse':
            quality = self._sse
        self.quality = quality
        self._score = None
    def _compute_random_centroids(self):
        """Compute a set of random centroids

        This method can be used to compute `self.k` centroids at random, and
        is the default calculation method for initialisation of the centroids

        :return: An k-length array of random items from self.X
        """
        idxs = np.random.choice(np.arange(len(self.X)), self.k, replace=False)
        print(f'idxs {idxs}')
        # TODO: Depends on self.X being pd.DataFrame
        return np.array([self.X.iloc[i] for i in idxs])

    def _init(self):
        """Initialise the centroids"""
        if self.init == 'random':
            self.init = self._compute_random_centroids
        self.centroids = self.init()
        return self.centroids

    def _kmeans(self):
        """Fit the optimal k-means clusters to the model"""
        def nearest_centroid(x):
            scores = []
            lowest = inf
            for k in self.centroids:
                dist = self.distance(x, k)
                scores.append(dist)
            return scores.index(min(scores))

        def has_converged(old, new):
            return np.all(np.isclose(old, new))

        scores = {}
        # Try n_init times to get the best centroids
        for i in range(self.n_init):
            print(f'Trying sample {i}...')
            self._init()
            new = None
            for j in range(self.max_iter):
                # Empty clusters for each centroid
                # Centroid will be added in later with distance 0 in next loop
                clusters = {c: [] for c, _ in enumerate(self.centroids)}
                if j % 10 == 0:
                    print(f'{j}/{self.max_iter}')
                old_centroids = self.centroids.copy()
                cluster_score = 0
                for k in range(len(self.X)):
                    x = self.X.iloc[k]
                    c = nearest_centroid(x)
                    clusters[c].append(x)
                # Update the centroids with the new data
                self.centroids = np.array([
                    np.array([
                        np.mean(np.array(clusters[i2])[:, j2])
                        for j2 in range(x.shape[0])
                    ])  # Get the mean for every column
                    for i2 in range(self.k)  # For every cluster
                ])
                if has_converged(old_centroids, self.centroids):
                    print(f'Converged after {j}')
                    break

            # Get the results for every cluster
            for centroid, cluster in clusters.items():
                cluster_score += self.quality(centroid, cluster)
            scores[cluster_score] = self.centroids
            print(scores)

        # Fit the best centroids to the class amd return self
        # TODO: Can we always assume it's min?
        self.centroids = scores[min(scores.keys())]

        return self

    def fit(self, X, _y=None):
        self.X = X
        # self._validate_params()

        # Early exit if we already computed the centroids
        if self.precomputed_centroids is not None:
            return self

        return self._kmeans()

    def predict(self, x):
        """Predict which cluster X should go in
        :param X: An array-like object with the same shape as a single object
        from self.X
        :return: The index of self.centroids which contains the correct cluster
        """
        dists = [self.distance(x, c) for c in self.centroids]
        return np.argmin(np.array(dists))

    def _sse(self, centroid, cluster):
        """Returns a value of quality metric.
        :param centroid: The index of self.centroids
        :param cluster: All the points in the cluster
        :return: The in-cluster SSE for the cluster `idx`
        """
        sse = 0
        mean = self.centroids[centroid]
        for c in cluster:
            d = self.distance(mean, c)
            sse += math.pow(d, 2)
        return sse

    def score(self, X, _y):
        """Calculate the score of the clustering algorithm

        Uses the sum of self.quality for each cluster, which defaults to SSE.
        In this case, the lower the score, the better
        """
        raise NotImplementedError("No need to calculate the score for the cluster."
                                  "Accuracy can be calcualted in KMeansClassifier.score")


class KMeansClassifier(KMeans, ClassifierMixin):
    """
    Subclass of the KMeans class, which allows classification type scores
    of X/y arrays, rather than calculating the overall score of the clustering
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels = []

    def count_clusters(self, X, y):
        cluster_counts = {i: Counter() for i, _ in enumerate(self.centroids)}
        for i in range(len(X)):
            x_i = X.iloc[i]
            y_i = y.iloc[i]
            cluster_idx = self.predict(x_i)
            cluster_counts[cluster_idx][y_i] += 1
        return cluster_counts

    def _infer(self, X, y):
        """Infer which clusters correspond to which labels

        Will find the cluster with the highest percentage of each output.
        TODO: This is very error-prone, as it could lead to some labels being
        overwritten, but is the best we have for now
        """
        cluster_counts = self.count_clusters(X, y)
        # Find the cluster with the highest amount of each y
        labels = {}
        for y_i in set(y):
            max_list = np.array([
                v[i]/sum(v.values())
                for c, v in cluster_counts.items()
                for i in v if i == y_i
            ])
            labels[np.argmax(max_list)] = y_i
            # Convert the labels dict into a list
        self.labels = [labels[i] for i in sorted(labels.keys())]
        print(f'cluster_counts: {cluster_counts}')
        print(f'labels: {labels}')

    def fit(self, X, y):
        """Assigns k clusters and uses labelled data to classify

        This method will first cluster the data using super.fit, and then use
        those clusters as the basis for the classification. It is assumed that
        self.k == amount of clusters in y.

        :param X: An array-like object with the features of the data
        :param y: A 1-D array-like object with the response variable
        :return: A fitted KMeansClassifier
        """
        super().fit(X)
        self._infer(X, y)

    def score(self, X, y):
        """Calculate the accuracy of the classification

        Uses the score method assigned to the class, defaults to accuracy
        """
        counter = Counter()
        for i in range(len(X)):
            pred = self.predict(X.iloc[i])
            actual = y.iloc[i]
            counter[pred == actual] += 1
        return counter[True] / sum(counter.values())
