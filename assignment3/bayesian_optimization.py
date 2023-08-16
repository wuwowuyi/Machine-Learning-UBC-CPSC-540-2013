import math
import unittest

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.stats import norm

"""
Implementation of acquisition functions, EI, GP-UCB and Thompson sampling are based on 
'A tutorial on Bayesian Optimization of Expensive Cost Functions,
with Application to Active User Modeling and Hierarchical Reinforcement Learning',
by Eric Brochu, Mike Cora and Nando de Freitas, 2009.
"""


class GP(object):
    """Gaussian Process"""

    def __init__(self, kernel, sigma, **kwargs):
        """
        Create a Gaussian prior.

        Args:
        - kernel: kernel function
        - sigma: standard deviation of the noise
        - kwargs: additional parameters for kernel function
        """
        self.kernel = kernel
        self.sigma = sigma
        self.kwargs = kwargs
        self.x_train = None
        self.y_train = None

    def add_data(self, x_train, y_train):
        """Add training data.

        Args:
        - x_train: training input (n, d) numpy array. n is # data points, d is #dimension
        - y_train: training targets, (n,) numpy array. n is # data points.
        """
        if x_train is None or y_train is None:
            raise ValueError('train data cannot be none.')

        if not isinstance(x_train, np.ndarray):
            x_train = np.array(x_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)

        if self.x_train is None:
            self.x_train = x_train
        else:
            self.x_train = np.concatenate((self.x_train, x_train.reshape(-1, 1)))

        if self.y_train is None:
            self.y_train = y_train
        else:
            self.y_train = np.concatenate((self.y_train, y_train))

    def posterior(self, x_test):
        """Compute the posterior mean and variance of x_test

        Args:
        - x_test: test input (n, d) 2-D numpy array

        Returns:
        - posterior mean (n, ) and deviation (n, ) of function output f(test)
        """
        y_mean = np.mean(self.y_train)
        y_train = self.y_train - y_mean
        n_train = y_train.shape[0]

        K_train = self.kernel(self.x_train, self.x_train, **self.kwargs)  # train variance (n_train, n_train)
        K_train_test = self.kernel(self.x_train, x_test, **self.kwargs)  # train and test covariance (n_train, n_test)
        K_test = self.kernel(x_test, x_test, **self.kwargs)  # test variance (n_test, n_test)

        # compute posterior mean mu
        # mu = k_*.T @ inv(K) @ y when mean(y) is zero. k_* is covariance of x_train and x_test.
        # Here we compute mu = (inv(L) @ k_*).T @ inv(L) @ y  where K = L @ L.T
        L = np.linalg.cholesky(K_train + self.sigma * np.eye(n_train))  # K = L @ L.T. L.shape=(n_train, n_train)
        Lk = np.linalg.solve(L, K_train_test)  # Lk= inv(L) @ K_train_test. Lk.shape=(n_train, n_test)
        mu = Lk.T @ np.linalg.solve(L, y_train)  # mu = Lk.T @ (inv(L) @ y). mu.shape=(n_test,)
        mu += y_mean
        # compute posterior variance
        # variance = K_test - k_*.T @ inv(K) @ k_*.T = K_test - (inv(L) @ k_*).T @ (inv(L) @ k_*.T)
        v = np.linalg.solve(L, K_train_test)  # v=inv(L) @ K_train_test. v.shape=(n_train, n_test)
        variance = np.diag(K_test) - np.diag(v.T @ v)
        return mu, np.sqrt(variance)

    def get_max(self):
        """Max of observed so far. """
        return np.max(self.y_train) if self.y_train is not None else -np.inf


def sqexp_kernel(x1, x2, ell=1.0, sf2=1.0):
    """Squared-exponential kernel.
    Also known as the Radial Basis Function kernel, the Gaussian kernel.
    k(x1, x2) = σ^2 * exp(−(x1 − x2)^2/2l^2)

    Args:
    - x1: (n, d) 2-D numpy array
    - x2: (m, d) 2-D numpy array
    - ell: the length scale parameter l^2 which determines the length of wiggles.
    - sf2: scale factor σ^2

    Returns:
    Covariance matrix of x1 and x2. shape=(n, m)
    """
    d = distance.cdist(x1, x2, 'sqeuclidean')
    return sf2 * np.exp(-0.5 * d / ell)


def gpei(gp, candidates, xi=0.1):
    """The EI(Expected Improvement) acquisition function for Bayesian optimization.

    Args:
    - gp: a Gaussian Process instance.
    - candidates: a set of data points

    Returns:
    A vector of index such that the highest index corresponds to the next point to select
    """
    y_max = gp.get_max()
    mu, s = gp.posterior(candidates)
    z_n = mu - y_max - xi
    z = z_n / s
    ei = z_n * norm.cdf(z) + s * norm.pdf(z)
    selected = np.argmax(ei)
    return candidates[selected], mu[selected], s[selected]


def gpucb(gp, candidates, t):
    """The GP-UCB acquisition function for Bayesian optimization.

    Args:
    - gp: a Gaussian Process instance.
    - candidates: a set of data points

    Returns:
    A vector of index such that the highest index corresponds to the next point to select
    """
    mu, s = gp.posterior(candidates)
    d = candidates.shape[1]
    beta_t = 2 * np.log((t ** (d/2 + 2)) * (math.pi ** 2) / 3 * s)
    ucb = mu + np.sqrt(beta_t) * s
    selected = np.argmax(ucb)
    return candidates[selected], mu[selected], s[selected]


def ts(gp, candidates):
    """Thompson Sampling
    Args:
    - gp: a Gaussian Process instance.
    - candidates: a set of data points

    Returns:
    A vector of index such that the highest index corresponds to the next point to select
    """
    mu, s = gp.posterior(candidates)
    samples = mu + s * np.random.randn(candidates.shape[0])
    selected = np.argmax(samples)
    return candidates[selected], mu[selected], s[selected]


def gpopt(f, gp, acq, **kwargs):
    """Bayesian optimization function.

    Args:
    - f: noisy function.
    - gp: a Gaussian Process instance
    - acq: acquisition function
    - candidates: some set of points we want to optimize at
    - kwargs: additional parameters for the acquisition function

    Returns:
    A sequence of points the algorithm thinks are optimal at every iteration. the last is the optimum.
    """
    iterations = 30
    x_range = 7
    selection, mean, sv = [], [], []
    for i in range(iterations):
        candidates = x_range * np.sort(np.random.rand(50)).reshape(-1, 1)
        if acq == 'ucb':
            x, x_mean, x_deviation = gpucb(gp, candidates, t=i+1)
        elif acq == 'ei':
            x, x_mean, x_deviation = gpei(gp, candidates, **kwargs)
        elif acq == 'ts':
            x, x_mean, x_deviation = ts(gp, candidates)
        selection.append(x)
        mean.append(x_mean)
        sv.append(x_deviation)
        gp.add_data(x, f(x))  # evaluate x and get a noisy observation, update gp.
    return selection, mean, sv


class BayesianOptimizationTest(unittest.TestCase):

    @staticmethod
    def f_sin(x):
        return (np.sin(x)).flatten()

    def _get_train(self, sigma, size, x_range=7):
        x_train = x_range * np.sort(np.random.rand(size)).reshape(-1, 1)
        y_train = self.f_sin(x_train) + sigma * np.random.randn(size)
        return x_train, y_train

    def _get_test(self, size, x_range=7):
        return x_range * np.sort(np.random.rand(size)).reshape(-1, 1)

    def test_fitting(self):
        """Test fitting a noisy sin function.
        """
        N = 10
        sigma = 0.01  # noise variance
        x_train, y_train = self._get_train(sigma, N)
        x_test = self._get_test(5*N)

        gp = GP(sqexp_kernel, sigma)
        gp.add_data(x_train, y_train)
        mu, deviation = gp.posterior(x_test)
        self._plot_GP_fitting(gp, x_test, mu, deviation)

    def test_BayesianOptimization(self):
        n = 3
        sigma = 0.01  # noise variance
        x_train, y_train = self._get_train(sigma, n)
        gp = GP(sqexp_kernel, sigma)
        gp.add_data(x_train, y_train)

        #candidates = self._get_test(500)
        noisy_sin = lambda x: self.f_sin(x) + sigma * np.random.randn(x.shape[0])
        optimals = gpopt(noisy_sin, gp, 'ucb')
        self._plot_bayesian_optimization(gp, optimals)


    def _plot_GP_fitting(self, gp, x_test, mu, deviation):
        # Plot the posterior distribution and some samples
        _, ax = plt.subplots()
        # Plot the distribution of the function (mean, covariance)
        ax.plot(gp.x_train, gp.y_train, 'bx', label='train data')  # train data, blue cross
        ax.plot(x_test, self.f_sin(x_test), 'r--', label='$sin(x)$')  # real function, red dashed line
        ax.plot(x_test, mu, 'k-', label='mean $\mu$')  # fitted function mean. black solid line
        ax.fill_between(x_test.flat, mu - deviation, mu + deviation, color='gray',
                         alpha=0.2, label='deviation $\sigma$')  # fitted function standard deviation
        ax.set_title(f'GP fitting a sin function. noise variance {gp.sigma:.2f}')
        ax.legend()
        plt.show()

    def _plot_bayesian_optimization(self, gp, optimals):
        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        x = self._get_test(50)
        ax1.plot(x, self.f_sin(x), 'r--', label='$sin(x)$')  # true f, red dashed line
        mu, s = gp.posterior(x)
        ax1.plot(x, mu, 'k-', label='$\mu_{p}$')  # posterior mean. black solid line
        ax1.fill_between(x.flat, mu - 2 * s, mu + 2 * s, color='gray',
                        alpha=0.2, label='$2 \sigma_{p}$')  # deviation.
        tested, mean, sv = optimals

        ax1.plot(gp.x_train, gp.y_train, 'bx')  # train data, blue cross
        #ax1.legend()
        iterations = len(tested)
        ax2.errorbar(range(iterations), self.f_sin(tested), sv)
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('objective')
        plt.show()
