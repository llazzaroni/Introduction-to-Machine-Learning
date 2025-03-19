import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared, WhiteKernel

from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import OneHotEncoder



def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """

    # Load training and test data and remove season
    train_df = pd.read_csv("train.csv").drop(['season'], axis=1)
    test_df = pd.read_csv("test.csv").drop(['season'], axis=1)

    # test set doesn't have CH, so place value of 0 for imputer
    # imputer won't attempt to replace 0
    test_df.insert(loc=1, column='price_CHF', value=0)

    train = train_df.to_numpy()
    test = test_df.to_numpy()

    # iterative imputer produces way more plausible results than SimpleImputer
    # important: use fit_transform for test set, performs way better than using imputer from train. doesn't make sense 2 me
    imp = IterativeImputer(max_iter=10, missing_values=np.nan)
    train = imp.fit_transform(train)
    test = imp.fit_transform(test)

    # split train data into X and y
    y_train = train[:,1] # slice the CH col
    X_train = np.delete(train, 1, axis=1)
    X_test = np.delete(test, 1, axis=1) # we are removing the dummy value for CHF 0 again here

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


if __name__ == "__main__":
    # Data loading
    X_train_all, y_train_all, X_test = data_loading()

    # normalize input data for kernels (expect mean 0, std 1)
    # don't refit for test split, keep consistency with train
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_all = scaler.fit_transform(X_train_all)
    X_test = scaler.transform(X_test)

    # split the training set further in a train and validation split, respecting chronological order
    split_index = int(len(X_train_all) * 0.8)
    X_train, y_train = X_train_all[:split_index], y_train_all[:split_index]
    X_val,   y_val   = X_train_all[split_index:], y_train_all[split_index:]

    kernel = ExpSineSquared(length_scale=10, periodicity=4) + RBF(length_scale=10) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1)

    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_val)

    plt.plot(np.arange(len(y_pred)), y_pred, label="y_pred")
    plt.plot(np.arange(len(y_val)), y_val, label="y_val")
    plt.legend()
    plt.show()

    # retrain, but this time with even more data
    gpr.fit(X_train_all, y_train_all)
    y_test = gpr.predict(X_test)

    # Save results in the required format
    dt = pd.DataFrame(y_test)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

