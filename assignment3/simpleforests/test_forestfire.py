import pandas as pd
import rf
from tools import *

days = ['mon', 'tue',  'wed', 'thu', 'fri', 'sat', 'sun']

months = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    ]


def load_data():
    # http://archive.ics.uci.edu/ml/datasets/Forest+Fires

    data = pd.read_csv('data/forestfires.csv')
    if data.isnull().values.any():
        data.dropna()

    data['month'] = data['month'].apply(lambda month: months.index(month))
    data['day'] = data['day'].apply(lambda day: days.index(day))
    data = data.apply(pd.to_numeric, downcast='float')

    X, Y = data.iloc[:, :-1], data.iloc[:, -1]  # last column is target
    Y = np.log(1+Y)  # instructions suggest this is easier to predict

    return X.to_numpy(), Y.to_numpy().reshape(-1, 1)


def run_test():
    X, Y = load_data()

    forest = rf.RegressionForest(X, Y)
    forest.fit(
        n_trees=30,
        max_depth=10,
        n_min_leaf=3,
        n_trials=4)
    Yhat = forest.predict(X)

    mse = np.mean((Y - Yhat)**2)
    print(mse)
    assert mse < 1.0

if __name__ == "__main__":
    run_test()

