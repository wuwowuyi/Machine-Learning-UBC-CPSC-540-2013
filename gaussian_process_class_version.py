import numpy as np
import matplotlib.pyplot as plt

# sampling

def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

n = 10
Xtest = np.linspace(-5, 5, n).reshape(-1, 1)
K_ = kernel(Xtest, Xtest)

L = np.linalg.cholesky(K_ + 1e-6 * np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n, 1)))

plt.plot(Xtest, f_prior)
plt.show()