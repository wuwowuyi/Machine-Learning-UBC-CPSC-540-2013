import numpy as np

# 7 MOVIES: Legally Blond; Matrix; Bourne Identity; You’ve Got Mail;
# The Devil Wears Prada; The Dark Knight; The Lord of the Rings.
# 9 users.
# preference matrix. Each row is a user's preference for movies.
P = np.array([[0, 0, -1, 0, -1, 1, 1],
              [-1, 1, 1, -1, 0, 1, 1],
              [0, 1, 1, 0, 0, -1, 1],
              [-1, 1, 1, 0, 0, 1, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [1, -1, 1, 1, 1, -1, 0],
              [-1, 1, -1, 0, -1, 0, 1],
              [0, -1, 0, 1, 1, -1, -1],
              [0, 0, -1, 1, 1, 0, -1]])

# Parameters
reg = 0.1 # regularization parameter
f = 2  # number of factors. (embedding has 2 dimensions)
m, n = P.shape

# Random Initialization. rand() samples from a uniform distribution over (0, 1)
X = 1 - 2 * np.random.rand(m, f)  # X is (m x f) in (-1, 1) range
Y = 1 - 2 * np.random.rand(f, n)  # Y is (f x n) in (-1, 1) range
X *= 0.1
Y *= 0.1

# Alternating Ridge Regression
for _ in range(100):
    X = np.linalg.solve(np.dot(Y, Y.T) + reg * np.eye(f), np.dot(Y, P.T)).T
    Y = np.linalg.solve(np.dot(X.T, X) + reg * np.eye(f), np.dot(X.T, P))
print('Alternating Ridge Regression:')
print(np.dot(X, Y))

# Re-initialize
X = 1 - 2 * np.random.rand(m, f)
Y = 1 - 2 * np.random.rand(f, n)
X *= 0.1
Y *= 0.1
# Alternating Weighted Ridge Regression
C = np.abs(P)  # Will be 0 only when P[i,j] == 0.
for _ in range(100):
    # Each user u has a different set of weights Cu
    for u, Cu in enumerate(C):
        Cu_diag = np.diag(Cu)
        A = Y @ Cu_diag @ Y.T + reg * np.eye(f)
        v = Y @ Cu_diag @ P[u]
        X[u] = np.linalg.solve(A, v).T  # solve Ax[u]=v by avoiding matrix inverse computation
    for i, Ci in enumerate(C.T):
        Ci_diag = np.diag(Ci)
        A = X.T @ Ci_diag @ X + reg * np.eye(f)
        v = X.T @ Ci_diag @ P[:, i]
        Y[:, i] = np.linalg.solve(A, v)
print('Alternating Weighted Ridge Regression:')
print(np.dot(X, Y))
