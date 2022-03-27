# Adapted from course provided builder.py
#
# Changes:
# - get oob (out of bag) samples for test
# - Use Abstract base class, etc.

from abc import ABC, abstractmethod

import numpy as np

from .builder import RegressionTreeBuilder
from .builder import ClassificationTreeBuilder


class Forest(ABC):

    def __init__(self, X, Y, sample_size=None):
        self.X, self.Y = X, Y
        self.sampled = set()  # index of data points sampled at least once
        self.sample_size = sample_size if sample_size else X.shape[0]

    def predict(self, X, return_tree_predictions=False):
        Ys = [tree.predict(X) for tree in self.trees]
        Y = 1./len(Ys) * sum(Ys)
        if return_tree_predictions:
            return Y, Ys
        else:
            return Y

    def fit(self,
            n_trees,
            **kwargs):

        self.trees = [
            self._tree_builder().fit(
                *self._bootstrap(self.X, self.Y),
                **kwargs)
            for _ in range(n_trees)
        ]
        return self

    def _bootstrap(self, X, Y):
        sample = np.random.randint(X.shape[0], size=self.sample_size)
        self.sampled.update(np.unique(sample))
        return X[sample], Y[sample]

    def get_oob_samples(self):
        all_samples = set(range(self.X.shape[0]))
        not_sampled = list(all_samples - self.sampled)
        if len(not_sampled) == 0:
            return None, None
        oob = np.array(not_sampled)
        return self.X[oob], self.Y[oob]

    @abstractmethod
    def _tree_builder(self):
        raise NotImplementedError


class RegressionForest(Forest):
    def _tree_builder(self):
        return RegressionTreeBuilder()

    def fit(self,
            n_trees,
            **kwargs):

        assert self.X.ndim == 2
        assert self.Y.ndim == 2
        assert self.Y.shape[1] == 1  # regression target must be a number.

        return super().fit(n_trees, **kwargs)


class ClassificationForest(Forest):
    def _tree_builder(self):
        return ClassificationTreeBuilder()

    def fit(self,
            n_trees,
            **kwargs):

        assert self.X.ndim == 2
        assert self.Y.ndim == 2

        return super().fit(n_trees, **kwargs)
