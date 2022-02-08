import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

data_file = 'prostate.data'
labels = np.loadtxt(data_file, max_rows=1, dtype=str)
X = np.loadtxt(data_file, skiprows=1)
y = X[:, -1]
X = X[:, :-1]
y_train, y_test = y[:50], y[50:]
X_train, X_test = X[:50], X[50:]

X_bar = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
y_bar = np.mean(y_train)  # y_bar subsumes theta_o
X_train -= X_bar
X_train /= X_std


def ridge(X, y, d2):
    """The ridge regression estimate of
    theta = (X^T * X + d2 * I)^-1 * X^T * y

    Args:
        - X: train data
        - y: value to predict
        - d2: ridge regularization
    Returns:
        - optimal theta
    """
    d = X.shape[1]  # d is #features
    t = np.linalg.inv(X.T @ X + d2 * np.eye(d))  # t.shape=(d, d)
    return np.dot(t @ X.T, y)  # theta.shape=(d, 1)


def train(k=100):
    """Compute the optimal theta for various regularization delta.
    Args:
        - k: number of delta to try
    Returns:
        - d2_candidates: delta tried. d2_candidates.shape=(k,)
        - estimates: best theta for each delta. estimates.shape=(k, d) where d is #attributes.
    """
    d2_candidates = np.sort(10 ** np.random.uniform(-1.5, 3.5, k))
    target = y_train - y_bar
    estimates = [ridge(X_train, target, d2) for d2 in d2_candidates]
    return d2_candidates, np.stack(estimates)


def plot_regularization_path(save=False):
    d2, estimates = train(k=1000)
    _, ax = plt.subplots()
    ax.plot(d2, estimates)
    ax.set_xscale('log')
    ax.set_xlabel('δ^2')
    ax.set_ylabel('θ')
    ax.legend(labels)
    ax.grid(True, linestyle='dashed')
    plt.show() if not save \
        else plt.savefig(f'reg_path_{datetime.now().strftime("%y-%m-%d-%H-%M")}.png')


def compute_error():
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


def plot_error(save=False):
    d2, t_error, train_error = compute_error()
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


if __name__ == '__main__':
    plot_regularization_path()
    plot_error()
