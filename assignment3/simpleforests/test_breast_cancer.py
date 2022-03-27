# Adapted from course provided test_breast_cancer.py.
#
# Changes:
# - Load and preprocess data using Pandas
# - test out of bagging accuracy.
#   Accuracy of the whole data set ranges from 0.96 to 0.98.
#   The oob accuracy ranges from 0.9 to 1 with 10 trees,
#   and most of the time 1.0 with 20 trees.


import pandas as pd

import rf
from tools import *


def load_data():
    # data downloaded from
    # https://archive-beta.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+original

    data = pd.read_csv('data/breast-cancer-wisconsin.data', na_values='?')
    data.dropna()
    # first column is user id, last is target
    X, y = data.iloc[:, 1:-1], data.iloc[:, -1]

    X = X.apply(pd.to_numeric, downcast='float')
    class_names = y.unique()
    y = y.apply(lambda v: 0 if v == 2 else 1)  # 2 for benign, 4 for malignant

    return X.to_numpy(), y.to_numpy(), class_names


def run_test():
    X, y, class_names = load_data()

    forest = rf.ClassificationForest(X, encode_one_of_n(y), X.shape[0] // 3)
    forest.fit(
        n_trees=10,
        max_depth=4,
        n_min_leaf=1,
        n_trials=4)

    Yhat = forest.predict(X)
    Yhat = max_of_n_prediction(Yhat)
    acc = np.mean(y == Yhat)
    print(f'accuracy is {acc}')
    assert acc > 0.95

    X_oob, Y_oob = forest.get_oob_samples()
    if X_oob is not None:
        Y_oob = decode_one_of_n(Y_oob)
        Y_oob_hat = forest.predict(X_oob)
        Y_oob_hat = max_of_n_prediction(Y_oob_hat)
        oob_acc = np.mean(Y_oob == Y_oob_hat)
        print(f'oob accuracy is {oob_acc}')


if __name__ == "__main__":
    run_test()
