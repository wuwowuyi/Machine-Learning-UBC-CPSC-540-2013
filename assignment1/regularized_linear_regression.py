"""
Assignment 1 question 1.

Regularized ridge regression.

Run the two unit tests to generate figures requested by the assignment question.
"""

import unittest

import numpy as np
import matplotlib.pyplot as plt


data_file = 'prostate.data'


def load_data(train_size=50):
    """
    Load training data.

    Args:
        - train_size: size of training data

    Returns:
        - X_train: training input. shape=(train_size, n) where n is number of features.
        - y_train: training targets. shape=(train_size,)
        - X_test: test input. shape=(M - train_size, n) where M is total size of data.
        - y_test: test targets. shape=(M - train_size,)
    """
    X = np.loadtxt(data_file, skiprows=1)  # the first row contains feature labels.
    y = X[:, -1]
    X = X[:, :-1]  # each row is a data point.
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    return X_train, y_train, X_test, y_test


def ridge(X, y, d2):
    """The optimal model parameters of ridge regression are:
    theta = (X.T * X + d2 * I)^-1 * X.T * y

    Assume m is the number of data points, n is the number of features.
    In this case we have n << m.

    Args:
        - X: train input data. X.shape= (m, n).
        - y: train target. y.shape=(m,)
        - d2: ridge regularization strength, is a real number.
    Returns:
        - optimal theta. shape=(n+1,)
    """
    m, n = X.shape
    X = np.c_[X, np.ones(m)]  # append an all one column for bias
    reg = d2 * np.eye(n + 1)
    reg[n, n] = 0  # do not regularize bias
    t = np.linalg.inv(X.T @ X + reg)
    return t @ (X.T @ y)


def train(X, y, k=100):
    """Compute the optimal theta for various regularization strength delta.
    Args:
        - X: input data. X.shape= (m, n).
        - y: target. y.shape=(m,)
        - k: number of delta to try
    Returns:
        - d2_candidates: delta tried. d2_candidates.shape=(k,)
        - estimates: best theta for each delta. shape=(k, n+1) where n is number of features
    """
    d2_candidates = np.sort(np.random.uniform(-1.5, 3.5, k))
    estimates = [ridge(X, y, 10 ** d2) for d2 in d2_candidates]
    return d2_candidates, np.stack(estimates)


def predict(X, y, y_bar, theta):
    """
    Predict target value, and returns the prediction error.

    Assume m is the number of input data points, n is the number of features,
    and k is the number of estimated models using different regularization strength.

    Args:
        - X: input data. shape=(m, n)
        - y: true target. shape=(m,)
        - y_bar: mean of training targets.
        - theta: estimated model parameters. shape=(k, n+1)

    Returns:
         - prediction error. shape=(k,)
    """
    m, _ = X.shape
    X = np.c_[X, np.ones(m)]
    y_hat = theta @ X.T + y_bar  # shape=(k, m)
    error = np.linalg.norm(y_hat - y, axis=1) / np.linalg.norm(y)
    return error


def plot_regularization_path(d2, estimates):
    """
    Plot the regularization path for each feature as we vary the regularization strength delta.

    Args:
        - d2: an array of regularization delta tried. shape=(k, )
        - estimates: best model parameters estimated. shape=(k, n) where n is number of features.
    """
    _, ax = plt.subplots()
    n = estimates.shape[1] - 1  # number of attributes
    for i in range(n):
        ax.plot(10 ** d2, estimates[:, i])
    ax.set_xscale('log')
    ax.set_xlabel(r'$\delta^2$')
    ax.set_ylabel('θ')
    labels = np.loadtxt(data_file, max_rows=1, dtype=str)
    ax.legend(labels[:-1])
    ax.grid(True, linestyle='dashed')
    plt.show()


def plot_error(d2, train_error, test_error):
    """
    Plot the relative prediction error.

    Args:
        - d2: an array of regularization delta tried. shape=(k, )
        - train_error: shape=(k,)
        - test_error: shape=(k,)
    """
    best = d2[np.argmin(test_error)]  # delta with the lowest test error.
    _, ax = plt.subplots()
    ax.plot(10 ** d2, train_error, color='blue', label='train')
    ax.plot(10 ** d2, test_error, color='green', label='test')
    ax.plot(best, np.min(test_error), 'g^')
    ax.set_xscale('log')
    ax.set_xlabel(f'$\delta^2$ best={best:.4f}')
    ax.set_ylabel('$||y - Xθ||_2|/||y||_2$')
    ax.grid(True, linestyle='dashed')
    ax.legend()
    plt.show()


class RegularizedLinearRegressionTest(unittest.TestCase):

    def setUp(self):
        self._load_data()
        self._normalize()

    def _load_data(self, train_size=50):
        self.X_train, self.y_train, self.X_test, self.y_test = load_data(train_size)

    def _normalize(self):
        self.X_bar = np.mean(self.X_train, axis=0)
        self.X_std = np.std(self.X_train, axis=0)
        self.y_bar = np.mean(self.y_train)
        self.X_train -= self.X_bar
        self.X_train /= self.X_std + 1e-8

    def test_plot_regularization_path(self):
        d2, estimates = train(self.X_train, self.y_train - self.y_bar)
        plot_regularization_path(d2, estimates)

    def test_plot_error(self):
        d2, estimates = train(self.X_train, self.y_train - self.y_bar)
        train_error = predict(self.X_train, self.y_train, self.y_bar, estimates)
        X_test = (self.X_test - self.X_bar) / self.X_std  # normalize test input
        test_error = predict(X_test, self.y_test, self.y_bar, estimates)
        plot_error(d2, train_error, test_error)
