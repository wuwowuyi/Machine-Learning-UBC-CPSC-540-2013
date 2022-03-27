# Adapted from course provided test_breast_cancer.py.
#
# Changes:
# - Load and preprocess data using Pandas
#
# Observations:
# - reduce number of features sampled (n_trials) and increase number of trees slightly increase accuracy
# - if bootstrap only sample a portion of the whole dataset, usually the accuracy is below 0.95.

import pandas as pd
import rf
from tools import *


def load_data():
    # data downloaded from
    # https://archive.ics.uci.edu/ml/datasets/ecoli

    data = pd.read_csv('data/ecoli.data', header=None, sep='\s+')
    # first column is a Sequence Name that has no predictive value, last is target
    X, y = data.iloc[:, 1:-1], data.iloc[:, -1]

    X = X.apply(pd.to_numeric, downcast='float')
    class_names = y.unique()
    names = list(class_names)
    y = y.apply(lambda v: names.index(v))

    return X.to_numpy(), y.to_numpy(), class_names

def run_test():
    X, Y, class_names = load_data()

    forest = rf.ClassificationForest(X, encode_one_of_n(Y), X.shape[0])
    forest.fit(
        n_trees=30,
        max_depth=7,
        n_min_leaf=1,
        n_trials=3)
    Yhat = forest.predict(X)
    Yhat = max_of_n_prediction(Yhat)
    acc = np.mean(Y == Yhat)
    print(acc)
    assert acc > 0.95

if __name__ == "__main__":
    run_test()

