import unittest

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

"""
Regularized ridge regression.

"""
data_file = 'prostate.data'


def load_data(train_size=50):
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
        - optimal theta. shape=(n,)
    """
    n = X.shape[1]
    t = np.linalg.inv(X.T @ X + d2 * np.eye(n))  # t.shape=(n, n)
    return t @ X.T.dot(y)


def train(X, y, k=100):
    """Compute the optimal theta for various regularization strength delta.
    Args:
        - k: number of delta to try
    Returns:
        - d2_candidates: delta tried. d2_candidates.shape=(k,)
        - estimates: best theta for each delta. shape=(k, n) where n is number of features
    """
    d2_candidates = np.sort(np.random.uniform(-1.5, 3.5, k))
    estimates = [ridge(X, y, 10 ** d2) for d2 in d2_candidates]
    return d2_candidates, np.stack(estimates)


def plot_regularization_path(d2, estimates):
    """
    Plot the regularization path for each feature as we vary the regularization strength delta.
    """
    _, ax = plt.subplots()
    n = estimates.shape[1]
    for i in range(n):
        ax.plot(10 ** d2, estimates[:, i])
    ax.set_xscale('log')
    ax.set_xlabel('δ^2')
    ax.set_ylabel('θ')
    labels = np.loadtxt(data_file, max_rows=1, dtype=str)
    ax.legend(labels)
    ax.grid(True, linestyle='dashed')
    plt.show()


def compute_error(X_train, y_train, X_test, y_test, X_bar, X_std, y_bar):
    """Compute prediction error and train error.
    """
    d2, estimates = train(1000)
    estimates = estimates[..., np.newaxis]  # estimates.shape=(k, d, 1)
    y_hat = y_bar + np.dot((X_test - X_bar) / X_std, estimates).squeeze()  # shape=(n, k) where n=#test, k=#theta
    y_train_hat = y_bar + np.dot(X_train, estimates).squeeze()  # shape=(m, k) where m=#train, k=#theta
    y1 = y_test[:, np.newaxis]
    t_error = np.linalg.norm(y1 - y_hat, axis=0) / np.linalg.norm(y1, axis=0)
    y2 = y_train[:, np.newaxis]
    train_error = np.linalg.norm(y2 - y_train_hat, axis=0) / np.linalg.norm(y2, axis=0)
    return d2, t_error, train_error


def plot_error(d2, t_error, train_error, save=False):
    best = d2[np.argmin(t_error)]  # delta with lowest test error.
    _, ax = plt.subplots()
    ax.plot(d2, t_error, color='green', label='test')
    ax.plot(best, np.min(t_error), 'g^')
    ax.plot(d2, train_error, color='blue', label='train')
    ax.set_xscale('log')
    ax.set_xlabel(f'δ^2. best={best:.4f}')
    ax.set_ylabel('||y - Xθ||/||y||')
    ax.grid(True, linestyle='dashed')
    ax.legend()
    plt.show() if not save \
        else plt.savefig(f'relative_error_{datetime.now().strftime("%y-%m-%d-%H-%M")}.png')


class RegularizedLinearRegressionTest(unittest.TestCase):

    def setUp(self):
        self.load_data()
        self.normalize()

    def load_data(self, train_size=50):
        self.X_train, self.y_train, self.X_test, self.y_test = load_data(train_size)

    def normalize(self):
        self.X_bar = np.mean(self.X_train, axis=0)
        self.X_std = np.std(self.X_train, axis=0)
        self.y_bar = np.mean(self.y_train)  # y_bar subsumes theta_o
        self.X_train -= self.X_bar
        self.X_train /= self.X_std

    def test_plot_regularization_path(self):
        d2, estimates = train(self.X_train, self.y_train - self.y_bar)
        plot_regularization_path(d2, estimates)

    def test_plot_error(self):
        d2, t_error, train_error = compute_error()
        plot_error(d2, t_error, train_error)
