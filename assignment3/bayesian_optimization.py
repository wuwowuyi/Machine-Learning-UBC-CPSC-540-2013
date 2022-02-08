import numpy as np
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

    def add_data(self, xtrain, ytrain):
        """Add training data.

        Args:
        - xtrain: (n, d) numpy array
        - ytrain: label, (n,) numpy array
        """
        self.train_mean = np.mean(ytrain) + self.sigma



    def posterior(self, xtest):
        """Compute the posterior mean and variance of xtest

        Args:
        - xtest: (n, d) numpy array

        Returns:
        posterior mean and variance of xtest
        """
        pass


def sqexp_kernel(x1, x2, ell=1.0, sf2=1.0):
    """Squared-exponential kernel.
    Also known as the Radial Basis Function kernel, the Gaussian kernel.
    k(x1, x2) = σ^2 * exp(−(x1 − x2)^2/2l^2)

    Args:
    - x1: (n, d) numpy array
    - x2: (m, d) numpy array
    - ell: the lengthscale parameter l^2 which determines the length of wiggles.
    - sf2: scale factor σ^2

    Returns:
    Covariance matrix (n, m) of x1 and x2
    """
    #d = np.linalg.norm(np.expand_dims(x1, axis=1) - x2, axis=2) ** 2
    d = distance.cdist(x1, x2, 'sqeuclidean')
    return sf2 * np.exp(-0.5 * d / ell)



def gpei(gp, candidates, **kwargs):
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
