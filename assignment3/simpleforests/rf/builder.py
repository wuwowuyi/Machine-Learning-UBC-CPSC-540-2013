# Adapted from course provided builder.py
#
# Changes:
# - implement _find_split_parameters() method
# - fix a number of errors when used for different datasets.


from abc import ABC, abstractmethod

import numpy as np

from .tree import InternalNode
from .tree import LeafNode


class TreeBuilder(ABC):

    def _find_split_parameters(self, X, Y, n_min_leaf, n_trials):
        """
        Compute parameters of the best split for the data X, Y.

        X: features, one data point per row
        Y: labels, one data point per row
        n_trials: the number of split dimensions to try.
        n_min_leaf: the minimum leaf size -- don't create a split with
            children smaller than this.

        Returns the pair (split_dim, split_threshold) or None if no appropriate
        split is found.  split_dim is an integer and split_threshold is a real
        number.

        Call self._information_gain(Y, Y_left, Y_right) to compute the
        information gain of a split.
        """
        m, n = X.shape  # m is # data points, n is # dimensions
        if m <= n_min_leaf:  # no more splits
            return None

        assert m == Y.shape[0]
        assert n >= n_trials

        choices = {}
        tried = set()
        for i in range(n_trials):
            dim = np.random.randint(n, size=1)[0]
            while dim in tried:
                dim = np.random.randint(n, size=1)[0]
            else:
                tried.add(dim)

            Xi = X[:, dim]
            x_values = np.unique(Xi)
            thresholds = (x_values[1:] + x_values[:-1]) / 2  # suppose all features are numeric
            max_gain, split = 0, None
            for t in thresholds:
                left_mask = Xi <= t
                right_mask = np.logical_not(left_mask)
                gain = self._information_gain(Y, Y[left_mask], Y[right_mask])
                if gain > max_gain:
                    max_gain = gain
                    split = t
            if max_gain > 0:
                choices[max_gain] = (dim, split)
        if len(choices.keys()) == 0:
            return None
        best_gain = sorted(choices.keys())[-1]
        return choices[best_gain]

    def fit(self, X, Y, max_depth, n_min_leaf, n_trials):
        yhat = Y.mean(axis=0).reshape(1, -1)

        # short circuit for pure leafs
        if np.all(Y == Y[0]):  # there is only one single value in Y
            return LeafNode(yhat)

        # avoid growing trees that are too deep
        if max_depth <= 0:
            return LeafNode(yhat)

        split_params = self._find_split_parameters(
                X, Y, n_min_leaf=n_min_leaf, n_trials=n_trials)

        # if we didn't find a good split point then become leaf
        if split_params is None:
            return LeafNode(yhat)

        split_dim, split_threshold = split_params

        mask_l = X[:, split_dim] < split_threshold
        mask_r = np.logical_not(mask_l)

        # cause error when n_min_leaf > 1
        # # refuse to make leafs that are too small
        # if np.sum(mask_l) < n_min_leaf or \
        #         np.sum(mask_r) < n_min_leaf:
        #     raise Exception("Leaf too small")

        # otherwise split this node recursively
        left_child = self.fit(
                X[mask_l],
                Y[mask_l],
                max_depth=max_depth - 1,
                n_min_leaf=n_min_leaf,
                n_trials=n_trials)

        right_child = self.fit(
                X[mask_r],
                Y[mask_r],
                max_depth=max_depth - 1,
                n_min_leaf=n_min_leaf,
                n_trials=n_trials)

        return InternalNode(
                dim=split_dim,
                threshold=split_threshold,
                left_child=left_child,
                right_child=right_child)

    @abstractmethod
    def _information_gain(self, y, y_l, y_r):
        raise NotImplementedError


class ClassificationTreeBuilder(TreeBuilder):

    def _entropy(self, x):
        x = x[x > 0]
        return -np.sum(x * np.log(x))

    def _information_gain(self, y, y_l, y_r):
        n = y.shape[0]
        n_l = y_l.shape[0]
        n_r = y_r.shape[0]

        H = self._entropy(y.mean(axis=0))  # mean is the prob. of each class.
        H_l = self._entropy(y_l.mean(axis=0))
        H_r = self._entropy(y_r.mean(axis=0))

        return H - n_l/n * H_l - n_r/n * H_r


class RegressionTreeBuilder(TreeBuilder):

    def _information_gain(self, y, y_l, y_r):
        assert y.size == y_l.size + y_r.size
        assert y_l.size > 0
        assert y_r.size > 0

        sse = np.sum((y - y.mean())**2)
        sse_l = np.sum((y_l - y_l.mean())**2)
        sse_r = np.sum((y_r - y_r.mean())**2)

        return sse - sse_l - sse_r

