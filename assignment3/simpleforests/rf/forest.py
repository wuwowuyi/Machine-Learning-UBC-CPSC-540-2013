# Adapted from course provided builder.py
#
# Changes:
# - get oob (out of bag) samples for test
# - Python 2 to 3. use Abstract base class, etc.
# - add comments

from abc import ABC, abstractmethod

import numpy as np

from .builder import RegressionTreeBuilder
from .builder import ClassificationTreeBuilder


class Forest(ABC):

    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.sampled = set()

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
        sample = np.random.randint(X.shape[0], size=X.shape[0] // 3)
        self.sampled.update(np.unique(sample))
        return X[sample], Y[sample]

    def get_oob_samples(self):
        all_samples = set(range(self.X.shape[0]))
        oob = np.array(list(all_samples - self.sampled))
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
