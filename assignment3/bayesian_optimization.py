import unittest

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance


class GP(object):
    """Gaussian Process"""

    def __init__(self, kernel, sigma, **kwargs):
        """
        Args:
        - kernel: kernel function
        - sigma: standard deviation of the noise
        - kwargs: additional parameters to be passed to the kernel
        """
        self.kernel = kernel
        self.sigma = sigma
        self.kwargs = kwargs

    def add_data(self, x_train, y_train):
        """Add training data.

        Args:
        - x_train: training input (n, d) numpy array
        - y_train: training targets, (n,) numpy array
        """
        self.x_train = x_train
        self.y_mean = np.mean(y_train)
        self.y_train = y_train - self.y_mean

    def posterior(self, test):
        """Compute the posterior mean and variance of xtest

        Args:
        - test: test input (n, d) 2-D numpy array

        Returns:
        posterior mean (n, ) and variance (n, n) of function output f(test)
        """
        n_train = self.y_train.shape[0]
        K_train = self.kernel(self.x_train, self.x_train, **self.kwargs)  # train variance (n_train, n_train)
        K_train_test = self.kernel(self.x_train, test, **self.kwargs)  # train and test covariance (n_train, n_test)
        K_test = self.kernel(test, test, **self.kwargs)  # test variance (n_test, n_test)
        # compute posterior mean mu
        L = np.linalg.cholesky(K_train + self.sigma * np.eye(n_train))  # K = L @ L.T. L.shape=(n_train, n_train)
        Lk = np.linalg.solve(L, K_train_test)  # Lk= inv(L) @ kernel(x_train, test). Lk.shape=(n_train, n_test)
        mu = Lk.T @ np.linalg.solve(L, self.y_train)  # mu = Lk.T @ (inv(L) @ y). mu.shape=(n_test,)
        mu += self.y_mean
        # compute posterior variance
        v = np.linalg.solve(L, K_train_test)  # v=inv(L) @ kernel(x_train, test). v.shape=(n_train, n_test)
        variance = K_test - v.T @ v  # variance.shape=(n_test, n_test)
        return mu, variance


def sqexp_kernel(x1, x2, ell=1.0, sf2=1.0):
    """Squared-exponential kernel.
    Also known as the Radial Basis Function kernel, the Gaussian kernel.
    k(x1, x2) = σ^2 * exp(−(x1 − x2)^2/2l^2)

    Args:
    - x1: (n, d) 2-D numpy array
    - x2: (m, d) 2-D numpy array
    - ell: the lengthscale parameter l^2 which determines the length of wiggles.
    - sf2: scale factor σ^2

    Returns:
    Covariance matrix of x1 and x2. shape=(n, m)
    """
    d = distance.cdist(x1, x2, 'sqeuclidean')
    return sf2 * np.exp(-0.5 * d / ell)


def gpei(gp, candidates, xi):
    """The EI(Expected Improvement) acquisition function for Bayesian optimization.

    Args:
    - gp: a Gaussian Process instance.
    - candidates: a set of data points

    Returns:
    A vector of index such that the highest index corresponds to the next point to select
    """
    pass


def gpucb(gp, candidates, **kwargs):
    """The GP-UCB acquisition function for Bayesian optimization.

    Args:
    - gp: a Gaussian Process instance.
    - candidates: a set of data points

    Returns:
    A vector of index such that the highest index corresponds to the next point to select
    """
    pass


def gpopt(f, gp, acq, candidates, **kwargs):
    """Bayesian optimization function.

    Args:
    - f: latent noisy function.
    - gp: a Gaussian Process instance
    - acq: acquisition function
    - candidates: some set of points we want to optimize at
    - kwargs: additional parameters for the acquisition function

    Returns:
    A sequence of points the algorithm thinks are optimal. the last is the optimum.
    """
    pass


class BayesianOptimizationTest(unittest.TestCase):

    def setUp(self):
        """
        Sample train and test points from a noisy sin function.
        """
        N = 10
        self.sigma = 0.01  # noise variance
        x_range = 7  # x ranges from 0 to 7
        self.x_train = x_range * np.sort(np.random.rand(N)).reshape(-1, 1)
        self.y_train = self.f_sin(self.x_train) + self.sigma * np.random.randn(N)
        self.x_test = x_range * np.sort(np.random.rand(5 * N)).reshape(-1, 1)

    @staticmethod
    def f_sin(x):
        return (np.sin(0.9 * x)).flatten()

    def test_GP(self):
        """Test fitting a noisy sin function.
        """
        gp = GP(sqexp_kernel, self.sigma)
        gp.add_data(self.x_train, self.y_train)
        mu, variance = gp.posterior(self.x_test)
        self._plot(mu, variance)

    def _plot(self, mu, variance):
        # Plot the postior distribution and some samples
        _, ax = plt.subplots()
        # Plot the distribution of the function (mean, covariance)
        ax.plot(self.x_train, self.y_train, 'bx', label='$(x_1, y_1)$')  # train data, blue cross
        ax.plot(self.x_test, self.f_sin(self.x_test), 'r--', label='$sin(x)$')  # real test, red dashed line
        ax.plot(self.x_test, mu, 'k-', label='$\mu_{p}$')  # posterior mean. black solid line
        sigma2 = np.sqrt(np.diag(variance))  # standard deviation
        ax.fill_between(self.x_test.flat, mu - 2 * sigma2, mu + 2 * sigma2, color='gray',
                         alpha=0.2, label='$2 \sigma_{p}$')  # varaince.
        ax.set_title(f'GP fitting a sin function. noise variance {self.sigma:.2f}')
        ax.legend()
        plt.show()


