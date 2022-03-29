import numpy as np
import pandas as pd

import rf
from tools import *


def load_data():
    # https://archive.ics.uci.edu/ml/datasets/wine+quality

    wq = pd.read_csv('data/winequality-red.csv', sep=';')
    wq = wq.apply(pd.to_numeric, downcast='float')
    x, y = wq.iloc[:, :-1], wq.iloc[:, -1]
    return x.to_numpy(), y.to_numpy().reshape(-1, 1)


def run_test():
    X, Y = load_data()

    forest = rf.RegressionForest(X, Y).fit(
            n_trees=10,
            max_depth=7,
            n_min_leaf=2,
            n_trials=5)
    Yhat = forest.predict(X)

    mse = np.mean((Y - Yhat)**2)
    print(mse)
    assert mse < 0.25

if __name__ == "__main__":
    run_test()

