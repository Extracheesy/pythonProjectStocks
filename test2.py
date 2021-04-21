#!/usr/bin/env python
# coding: utf-8

# # Objective
# * 20191002:
#     * Given prices for the last N days, we do prediction for the next N+H days, where H is the forecast horizon
#     * Using xgboost
# * 20191004 - Diff from StockPricePrediction_v6_xgboost.ipynb:
#     * Instead of using mean and std from train set to do scaling/unscaling, use mean and std from the last N days to do scaling/unscaling
# * 20191007 - Diff from StockPricePrediction_v6a_xgboost.ipynb:
#     * Include a validation set to do hyperparameter tuning
# * 20191018 - Diff from StockPricePrediction_v6b_xgboost.ipynb:
#     * Instead of tuning N, we use a fixed N and observe the results
# * 20191021 - Diff from StockPricePrediction_v6c_xgboost.ipynb:
#     * Instead of using only features about price, introduce more features and observe the results

print("# In[1069]:")

import chart_studio.plotly as py
import math
import matplotlib
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objs as go
import time
import os

import orca

from collections import defaultdict
from datetime import date
# from fastai.tabular import add_datepart
# from fastai.tabular import *
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
import re

# get_ipython().run_line_magic('matplotlib', 'inline')

# py.sign_in('<your-user-id>', '<your-api-key>') # sign in to plotly if you haven't done so

#### Input params ##################
folder = "./data/"
filename = "VTI_20130102_20181231.csv"

# Predicting on day 1008, date 2017-01-03 00:00:00
# Predicting on day 1050, date 2017-03-06 00:00:00
# Predicting on day 1092, date 2017-05-04 00:00:00
# Predicting on day 1134, date 2017-07-05 00:00:00
# Predicting on day 1176, date 2017-09-01 00:00:00
# Predicting on day 1218, date 2017-11-01 00:00:00
# Predicting on day 1260, date 2018-01-03 00:00:00
# Predicting on day 1302, date 2018-03-06 00:00:00
# Predicting on day 1344, date 2018-05-04 00:00:00
# Predicting on day 1386, date 2018-07-05 00:00:00
# Predicting on day 1428, date 2018-09-04 00:00:00
# Predicting on day 1470, date 2018-11-01 00:00:00


pred_day_list = [1008, 1050, 1092, 1134, 1176, 1218, 1260, 1302, 1344, 1386, 1428, 1470]

pred_day = 1008  # Predict for this day, for the next H-1 days. Note indexing of days start from 0.

H = 21  # Forecast horizon, in days. Note there are about 252 trading days in a year
train_size = 252 * 3  # Use 3 years of data as train set. Note there are about 252 trading days in a year
val_size = 252  # Use 1 year of data as validation set
# N = 21                         # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
N = 10  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features

n_estimators = 100  # Number of boosted trees to fit. default = 100
max_depth = 3  # Maximum tree depth for base learners. default = 3
learning_rate = 0.1  # Boosting learning rate (xgb’s “eta”). default = 0.1
min_child_weight = 1  # Minimum sum of instance weight(hessian) needed in a child. default = 1
subsample = 1  # Subsample ratio of the training instance. default = 1
colsample_bytree = 1  # Subsample ratio of columns when constructing each tree. default = 1
colsample_bylevel = 1  # Subsample ratio of columns for each split, in each level. default = 1
gamma = 0  # Minimum loss reduction required to make a further partition on a leaf node of the tree. default=0

model_seed = 100

fontsize = 14
ticklabelsize = 14

# Plotly colors
colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]

####################################

train_val_size = train_size + val_size  # Size of train+validation set
print("No. of days in train+validation set = " + str(train_val_size))

print("# In[1070]:")

tic1 = time.time()

# # Common functions

print("# In[1071]:")


def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)


def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a


def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    # Pandas removed `dt.week` in v1.1.10
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower()) if n != 'Week' else week
    mask = ~field.isna()
    df[prefix + 'Elapsed'] = np.where(mask, field.values.astype(np.int64) // 10 ** 9, np.nan)
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_mae(a, b):
    """
    Comp mean absolute error e_t = E[|a_t - b_t|]. a and b can be lists.
    Returns a vector of len = len(a) = len(b)
    """
    return np.mean(abs(np.array(a) - np.array(b)))


def get_rmse(a, b):
    """
    Comp RMSE. a and b can be lists.
    Returns a scalar.
    """
    return math.sqrt(np.mean((np.array(a) - np.array(b)) ** 2))


print("# In[1072]:")


def get_mov_avg_std(df, col, N):
    """
    Given a dataframe, get mean and std dev at timestep t using values from t-1, t-2, ..., t-N.
    Inputs
        df         : dataframe. Can be of any length.
        col        : name of the column you want to calculate mean and std dev
        N          : get mean and std dev at timestep t using values from t-1, t-2, ..., t-N
    Outputs
        df_out     : same as df but with additional column containing mean and std dev
    """
    mean_list = df[col].rolling(window=N, min_periods=1).mean()  # len(mean_list) = len(df)
    std_list = df[col].rolling(window=N, min_periods=1).std()  # first value will be NaN, because normalized by N-1

    # Add one timestep to the predictions
    mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
    std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))

    # Append mean_list to df
    df_out = df.copy()
    df_out[col + '_mean'] = mean_list
    df_out[col + '_std'] = std_list

    return df_out


def do_scaling(df, N):
    """
    Do scaling for the adj_close and lag cols
    """
    df.loc[:, 'adj_close_scaled'] = (df['adj_close'] - df['adj_close_mean']) / df['adj_close_std']
    for n in range(N, 0, -1):
        df.loc[:, 'adj_close_scaled_lag_' + str(n)] = (df['adj_close_lag_' + str(n)] - df['adj_close_mean']) / df[
            'adj_close_std']

        # Remove adj_close_lag column which we don't need anymore
        df.drop(['adj_close_lag_' + str(n)], axis=1, inplace=True)

    return df


def get_accuracy_trend(y_test, pred):
    y_pred = pd.DataFrame(pred)
    y_test = y_test.reset_index(drop=True)
    len_y_test = len(y_test)
    len_y_pred = len(y_pred)

    # y_test_target_raw = (y_test.shift(-1) / y_test) - 1
    y_test_target_raw = y_test.shift(-1) - y_test
    y_test_target_raw[y_test_target_raw > 0] = 1
    y_test_target_raw[y_test_target_raw <= 0] = 0

    y_test_target_raw.drop([len_y_test - 1], axis=0, inplace=True)

    # y_pred_target_raw = (y_pred.shift(-1) / y_pred) - 1
    y_pred_target_raw = y_pred.shift(-1) - y_pred
    y_pred_target_raw[y_pred_target_raw > 0] = 1
    y_pred_target_raw[y_pred_target_raw <= 0] = 0

    y_pred_target_raw.drop([len_y_pred - 1], axis=0, inplace=True)

    len_y_pred = len(y_pred_target_raw)

    accuracy = round(accuracy_score(y_test_target_raw, y_pred_target_raw, normalize=False) / len_y_pred, 6)

    # print("==========> accuracy: ",accuracy)

    return accuracy


def pred_xgboost(model, X_test_ex_adj_close, N, H, prev_vals, prev_mean_val, prev_std_val):
    """
    Do recursive forecasting using xgboost
    Inputs
        model              : the xgboost model
        X_test_ex_adj_close: features of the test set, excluding adj_close_scaled values
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        prev_vals          : numpy array. If predict at time t,
                             prev_vals will contain the N unscaled values at t-1, t-2, ..., t-N
        prev_mean_val      : the mean of the unscaled values at t-1, t-2, ..., t-N
        prev_std_val       : the std deviation of the unscaled values at t-1, t-2, ..., t-N
    Outputs
        Times series of predictions. Numpy array of shape (H,). This is unscaled.
    """
    forecast = prev_vals.copy()

    for n in range(H):
        forecast_scaled = (forecast[-N:] - prev_mean_val) / prev_std_val

        # Create the features dataframe
        X = X_test_ex_adj_close[n:n + 1].copy()
        for n in range(N, 0, -1):
            X.loc[:, "adj_close_scaled_lag_" + str(n)] = forecast_scaled[-n]

        # Do prediction
        est_scaled = model.predict(X)

        # Unscale the prediction
        forecast = np.concatenate([forecast,
                                   np.array((est_scaled * prev_std_val) + prev_mean_val).reshape(1, )])

        # Comp. new mean and std
        prev_mean_val = np.mean(forecast[-N:])
        prev_std_val = np.std(forecast[-N:])

    return forecast[-H:]


def train_pred_eval_model(X_train_scaled,
                          y_train_scaled,
                          X_test_ex_adj_close,
                          y_test,
                          N,
                          H,
                          prev_vals,
                          prev_mean_val,
                          prev_std_val,
                          seed=100,
                          n_estimators=100,
                          max_depth=3,
                          learning_rate=0.1,
                          min_child_weight=1,
                          subsample=1,
                          colsample_bytree=1,
                          colsample_bylevel=1,
                          gamma=0):
    '''
    Train model, do prediction, scale back to original range and do evaluation
    Use XGBoost here.
    Inputs
        X_train_scaled     : features for training. Scaled to have mean 0 and variance 1
        y_train_scaled     : target for training. Scaled to have mean 0 and variance 1
        X_test_ex_adj_close: features of the test set, excluding adj_close_scaled values
        y_test             : target for test. Actual values, not scaled.
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        prev_vals          : numpy array. If predict at time t,
                             prev_vals will contain the N unscaled values at t-1, t-2, ..., t-N
        prev_mean_val      : the mean of the unscaled values at t-1, t-2, ..., t-N
        prev_std_val       : the std deviation of the unscaled values at t-1, t-2, ..., t-N
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :
    Outputs
        rmse               : root mean square error of y_test and est
        mape               : mean absolute percentage error of y_test and est
        mae                : mean absolute error of y_test and est
        est                : predicted values. Same length as y_test
    '''

    model = XGBRegressor(objective='reg:squarederror',
                         seed=model_seed,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel,
                         gamma=gamma)

    # Train the model
    model.fit(X_train_scaled, y_train_scaled)

    # Get predicted labels and scale back to original range
    est = pred_xgboost(model, X_test_ex_adj_close, N, H, prev_vals, prev_mean_val, prev_std_val)

    # Calculate RMSE, MAPE, MAE
    rmse = get_rmse(y_test, est)
    mape = get_mape(y_test, est)
    mae = get_mae(y_test, est)
    accuracy = get_accuracy_trend(y_test, est)

    return rmse, mape, mae, accuracy, est, model.feature_importances_


def add_lags(df, N, lag_cols):
    """
    Add lags up to N number of days to use as features
    The lag columns are labelled as 'adj_close_lag_1', 'adj_close_lag_2', ... etc.
    """
    # Use lags up to N number of days to use as features
    df_w_lags = df.copy()
    df_w_lags.loc[:, 'order_day'] = [x for x in list(
        range(len(df)))]  # Add a column 'order_day' to indicate the order of the rows by date
    merging_keys = ['order_day']  # merging_keys
    shift_range = [x + 1 for x in range(N)]
    for shift in shift_range:
        train_shift = df_w_lags[merging_keys + lag_cols].copy()

        # E.g. order_day of 0 becomes 1, for shift = 1.
        # So when this is merged with order_day of 1 in df_w_lags, this will represent lag of 1.
        train_shift['order_day'] = train_shift['order_day'] + shift

        foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
        train_shift = train_shift.rename(columns=foo)

        df_w_lags = pd.merge(df_w_lags, train_shift, on=merging_keys, how='left')  # .fillna(0)
    del train_shift

    return df_w_lags


def get_error_metrics(df,
                      train_size,
                      N,
                      H,
                      seed=100,
                      n_estimators=100,
                      max_depth=3,
                      learning_rate=0.1,
                      min_child_weight=1,
                      subsample=1,
                      colsample_bytree=1,
                      colsample_bylevel=1,
                      gamma=0):
    """
    Given a series consisting of both train+validation, do predictions of forecast horizon H on the validation set,
    at H/2 intervals.
    Inputs
        df                 : train + val dataframe. len(df) = train_size + val_size
        train_size         : size of train set
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :

    Outputs
        mean of rmse, mean of mape, mean of mae, dictionary of predictions
    """
    rmse_list = []  # root mean square error
    mape_list = []  # mean absolute percentage error
    mae_list = []  # mean absolute error
    accuracy_list = []  # accuracy absolute error
    preds_dict = {}

    # Add lags up to N number of days to use as features
    df = add_lags(df, N, ['adj_close'])

    # Get mean and std dev at timestamp t using values from t-1, ..., t-N
    df = get_mov_avg_std(df, 'adj_close', N)

    # Do scaling
    df = do_scaling(df, N)

    # Get list of features
    features_ex_adj_close = [
        'year',
        'month',
        'week',
        'day',
        'dayofweek',
        'dayofyear',
        'is_month_end',
        'is_month_start',
        'is_quarter_end',
        'is_quarter_start',
        'is_year_end'
    ]
    features = features_ex_adj_close  # features contain all features, including adj_close_lags
    for n in range(N, 0, -1):
        features.append("adj_close_scaled_lag_" + str(n))

    for i in range(train_size, len(df) - H + 1, int(H / 2)):
        # Split into train and test
        train = df[i - train_size:i].copy()
        test = df[i:i + H].copy()

        # Drop the NaNs in train
        train.dropna(axis=0, how='any', inplace=True)

        # Split into X and y
        X_train_scaled = train[features]
        y_train_scaled = train['adj_close_scaled']
        X_test_ex_adj_close = test[features_ex_adj_close]
        y_test = test['adj_close']
        prev_vals = train[-N:]['adj_close'].to_numpy()
        prev_mean_val = test.iloc[0]['adj_close_mean']
        prev_std_val = test.iloc[0]['adj_close_std']

        rmse, mape, mae, accuracy, est, _ = train_pred_eval_model(X_train_scaled,
                                                                  y_train_scaled,
                                                                  X_test_ex_adj_close,
                                                                  y_test,
                                                                  N,
                                                                  H,
                                                                  prev_vals,
                                                                  prev_mean_val,
                                                                  prev_std_val,
                                                                  seed=seed,
                                                                  n_estimators=n_estimators,
                                                                  max_depth=max_depth,
                                                                  learning_rate=learning_rate,
                                                                  min_child_weight=min_child_weight,
                                                                  subsample=subsample,
                                                                  colsample_bytree=colsample_bytree,
                                                                  colsample_bylevel=colsample_bylevel,
                                                                  gamma=gamma)
        #         print("N = " + str(N) + ", i = " + str(i) + ", rmse = " + str(rmse) + ", mape = " + str(mape) + ", mae = " + str(mae))

        rmse_list.append(rmse)
        mape_list.append(mape)
        mae_list.append(mae)
        accuracy_list.append(accuracy)
        preds_dict[i] = est

    return np.mean(rmse_list), np.mean(mape_list), np.mean(mae_list), np.mean(accuracy_list), preds_dict


def get_error_metrics_one_pred(df,
                               train_size,
                               N,
                               H,
                               seed=100,
                               n_estimators=100,
                               max_depth=3,
                               learning_rate=0.1,
                               min_child_weight=1,
                               subsample=1,
                               colsample_bytree=1,
                               colsample_bylevel=1,
                               gamma=0):
    """
    Given a series consisting of both train+test, do one prediction of forecast horizon H on the test set.
    Inputs
        df                 : train + test dataframe. len(df) = train_size + test_size
        train_size         : size of train set
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :

    Outputs
        rmse, mape, mae, predictions
    """
    # Add lags up to N number of days to use as features
    df = add_lags(df, N, ['adj_close'])

    # Get mean and std dev at timestamp t using values from t-1, ..., t-N
    df = get_mov_avg_std(df, 'adj_close', N)

    # Do scaling
    df = do_scaling(df, N)

    # Get list of features
    features_ex_adj_close = [
        'year',
        'month',
        'week',
        'day',
        'dayofweek',
        'dayofyear',
        'is_month_end',
        'is_month_start',
        'is_quarter_end',
        'is_quarter_start',
        'is_year_end'
    ]
    features = features_ex_adj_close  # features contain all features, including adj_close_lags
    for n in range(N, 0, -1):
        features.append("adj_close_scaled_lag_" + str(n))

    # Split into train and test
    train = df[:train_size].copy()
    test = df[train_size:train_size + H].copy()

    # Drop the NaNs in train
    train.dropna(axis=0, how='any', inplace=True)

    # Split into X and y
    X_train_scaled = train[features]
    y_train_scaled = train['adj_close_scaled']
    X_test_ex_adj_close = test[features_ex_adj_close]
    y_test = test['adj_close']
    prev_vals = train[-N:]['adj_close'].to_numpy()
    prev_mean_val = test.iloc[0]['adj_close_mean']
    prev_std_val = test.iloc[0]['adj_close_std']

    rmse, mape, mae, accuracy, est, feature_importances = train_pred_eval_model(X_train_scaled,
                                                                                y_train_scaled,
                                                                                X_test_ex_adj_close,
                                                                                y_test,
                                                                                N,
                                                                                H,
                                                                                prev_vals,
                                                                                prev_mean_val,
                                                                                prev_std_val,
                                                                                seed=seed,
                                                                                n_estimators=n_estimators,
                                                                                max_depth=max_depth,
                                                                                learning_rate=learning_rate,
                                                                                min_child_weight=min_child_weight,
                                                                                subsample=subsample,
                                                                                colsample_bytree=colsample_bytree,
                                                                                colsample_bylevel=colsample_bylevel,
                                                                                gamma=gamma)

    return rmse, mape, mae, accuracy, est, feature_importances, features


def train_pred_eval_model_GS(X_train_scaled,
                             y_train_scaled,
                             X_test_ex_adj_close,
                             y_test,
                             N,
                             H,
                             prev_vals,
                             prev_mean_val,
                             prev_std_val,
                             seed=100,
                             n_estimators=100,
                             max_depth=3,
                             learning_rate=0.1,
                             min_child_weight=1,
                             subsample=1,
                             colsample_bytree=1,
                             colsample_bylevel=1,
                             gamma=0):
    '''
    Train model, do prediction, scale back to original range and do evaluation
    Use XGBoost here.
    Inputs
        X_train_scaled     : features for training. Scaled to have mean 0 and variance 1
        y_train_scaled     : target for training. Scaled to have mean 0 and variance 1
        X_test_ex_adj_close: features of the test set, excluding adj_close_scaled values
        y_test             : target for test. Actual values, not scaled.
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        prev_vals          : numpy array. If predict at time t,
                             prev_vals will contain the N unscaled values at t-1, t-2, ..., t-N
        prev_mean_val      : the mean of the unscaled values at t-1, t-2, ..., t-N
        prev_std_val       : the std deviation of the unscaled values at t-1, t-2, ..., t-N
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :
    Outputs
        rmse               : root mean square error of y_test and est
        mape               : mean absolute percentage error of y_test and est
        mae                : mean absolute error of y_test and est
        est                : predicted values. Same length as y_test
    '''

    model = XGBRegressor(objective='reg:squarederror',
                         seed=model_seed,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel,
                         gamma=gamma)

    params = {
        'polynomialfeatures__degree': [2, 3],
        # 'selectkbest__k': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 ,19 ,20, 21],
    }
    model_XGB = make_pipeline(PolynomialFeatures(2, include_bias=False),
                              # SelectKBest(f_classif, k=21),
                              model)
    print("GridSearch...")
    Classifier_XGB = GridSearchCV(model_XGB, param_grid=params, cv=4)
    model = Classifier_XGB
    print("GridSearch completed")

    # Train the model
    model.fit(X_train_scaled, y_train_scaled)

    print("model :", model)
    print("best_param: ", model.best_params_)
    print("best_score: ", model.best_score_)

    # Get predicted labels and scale back to original range
    est = pred_xgboost(model, X_test_ex_adj_close, N, H, prev_vals, prev_mean_val, prev_std_val)

    # Calculate RMSE, MAPE, MAE
    rmse = get_rmse(y_test, est)
    mape = get_mape(y_test, est)
    mae = get_mae(y_test, est)
    accuracy = get_accuracy_trend(y_test, est)

    return rmse, mape, mae, accuracy, est


def get_error_metrics_GS(df,
                         train_size,
                         N,
                         H,
                         seed=100,
                         n_estimators=100,
                         max_depth=3,
                         learning_rate=0.1,
                         min_child_weight=1,
                         subsample=1,
                         colsample_bytree=1,
                         colsample_bylevel=1,
                         gamma=0):
    """
    Given a series consisting of both train+test, do one prediction of forecast horizon H on the test set.
    Inputs
        df                 : train + test dataframe. len(df) = train_size + test_size
        train_size         : size of train set
        N                  : for feature at day t, we use lags from t-1, t-2, ..., t-N as features
        H                  : forecast horizon
        seed               : model seed
        n_estimators       : number of boosted trees to fit
        max_depth          : maximum tree depth for base learners
        learning_rate      : boosting learning rate (xgb’s “eta”)
        min_child_weight   : minimum sum of instance weight(hessian) needed in a child
        subsample          : subsample ratio of the training instance
        colsample_bytree   : subsample ratio of columns when constructing each tree
        colsample_bylevel  : subsample ratio of columns for each split, in each level
        gamma              :

    Outputs
        rmse, mape, mae, predictions
    """
    # Add lags up to N number of days to use as features
    df = add_lags(df, N, ['adj_close'])

    # Get mean and std dev at timestamp t using values from t-1, ..., t-N
    df = get_mov_avg_std(df, 'adj_close', N)

    # Do scaling
    df = do_scaling(df, N)

    # Get list of features
    features_ex_adj_close = [
        'year',
        'month',
        'week',
        'day',
        'dayofweek',
        'dayofyear',
        'is_month_end',
        'is_month_start',
        'is_quarter_end',
        'is_quarter_start',
        'is_year_end'
    ]
    features = features_ex_adj_close  # features contain all features, including adj_close_lags
    for n in range(N, 0, -1):
        features.append("adj_close_scaled_lag_" + str(n))

    # Split into train and test
    train = df[:train_size].copy()
    test = df[train_size:train_size + H].copy()

    # Drop the NaNs in train
    train.dropna(axis=0, how='any', inplace=True)

    # Split into X and y
    X_train_scaled = train[features]
    y_train_scaled = train['adj_close_scaled']
    X_test_ex_adj_close = test[features_ex_adj_close]
    y_test = test['adj_close']
    prev_vals = train[-N:]['adj_close'].to_numpy()
    prev_mean_val = test.iloc[0]['adj_close_mean']
    prev_std_val = test.iloc[0]['adj_close_std']

    rmse, mape, mae, accuracy, est = train_pred_eval_model_GS(X_train_scaled,
                                                              y_train_scaled,
                                                              X_test_ex_adj_close,
                                                              y_test,
                                                              N,
                                                              H,
                                                              prev_vals,
                                                              prev_mean_val,
                                                              prev_std_val,
                                                              seed=seed,
                                                              n_estimators=n_estimators,
                                                              max_depth=max_depth,
                                                              learning_rate=learning_rate,
                                                              min_child_weight=min_child_weight,
                                                              subsample=subsample,
                                                              colsample_bytree=colsample_bytree,
                                                              colsample_bylevel=colsample_bylevel,
                                                              gamma=gamma)

    return rmse, mape, mae, accuracy, est


def remove_row(df,row):
    df.drop([row], axis=0, inplace=True)
    return df


def set_param_row(pred_day, N, H, RMSE, MAPE, MAE, ACCURACY, n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, colsample_bylevel, gamma):

    row = []

    row.append(pred_day)
    row.append(N)
    row.append(H)
    row.append(RMSE)
    row.append(MAPE)
    row.append(MAE)
    row.append(ACCURACY)
    row.append(n_estimators)
    row.append(max_depth)
    row.append(learning_rate)
    row.append(min_child_weight)
    row.append(subsample)
    row.append(colsample_bytree)
    row.append(colsample_bylevel)
    row.append(gamma)

    return row

def addRow(df,ls):
    """
    Given a dataframe and a list, append the list as a new row to the dataframe.

    :param df: <DataFrame> The original dataframe
    :param ls: <list> The new row to be added
    :return: <DataFrame> The dataframe with the newly appended row
    """

    numEl = len(ls)

    newRow = pd.DataFrame(np.array(ls).reshape(1,numEl), columns = list(df.columns))

    df = df.append(newRow, ignore_index=True)

    return df

# Plot with plotly
def plot_df(df, x_col, y_col, x_label, y_label, legend, title, file_name):
    # Plot adjusted close over time
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8

    ax = df.plot(x=x_col, y=y_col, style='b-', grid=True)
    if (len(legend) > 0):
        ax.legend(['train_scaled'])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_title(title)

    fig = ax.get_figure()
    fig.savefig(file_name)

# # Load data

print("# In[1073]:")

df = pd.read_csv(folder + filename, sep=",")

# Convert Date column to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Change all column headings to be lower case, and remove spacing
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

# Sort by datetime
df.sort_values(by='date', inplace=True, ascending=True)

df.head(10)

print("# In[1074]:")

# Remove columns which you can't use as features
df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)

print("# In[1075]:")

plot_df(df, 'date', 'adj_close', 'date', 'USD', [], "no title", './figure/test_v6_1.pdf')

"""
data = [go.Scatter(
            x = df['date'],
            y = df['adj_close'],
            mode = 'lines')]

layout = dict(xaxis = dict(title = 'date'),
              yaxis = dict(title = 'USD'))

fig = dict(data=data, layout=layout)
#py.iplot(fig, filename=filename)
#fig.write_image("./figure/test_v6_1.pdf")
"""
# # Feature Engineering

print("# In[1076]:")

# create features
add_datepart(df, 'date', drop=False)
df.drop('Elapsed', axis=1, inplace=True)  # don't need this
df.head(50)

print("# In[1077]:")

# Change all column headings to be lower case, and remove spacing
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

print("# In[1078]:")

# Convert year to categorical feature, based on alphabetical order
df.loc[:, 'year'] = LabelEncoder().fit_transform(df['year'])
df[15:25]

# # EDA

print("# In[1079]:")

# Count number of nulls for each column
df.isnull().sum(axis=0)

print("# In[1080]:")

# Compute the average price for each month
avg_price_mth = df.groupby("month").agg({'adj_close': 'mean'}).reset_index()

# Plot

plot_df(avg_price_mth, 'month', 'adj_close', 'month', 'average adjusted closing price (USD)', [], "no title",
        './figure/test_v6_month.pdf')

"""
data = [go.Scatter(
            x = avg_price_mth['month'],
            y = avg_price_mth['adj_close'],
            mode = 'markers',
            marker=dict(
                color='LightSkyBlue',
                size=15,
                line=dict(
                color='MediumPurple',
                width=2
                ))
        )]

layout = dict(xaxis = dict(title = 'month'),
              yaxis = dict(title = 'average adjusted closing price (USD)'))

fig = dict(data=data, layout=layout)
#py.iplot(fig, filename='StockPricePrediction_v6d_avg_price_mth')
#fig.write_image("./figure/test_v6_month.pdf")
"""
print("# In[1081]:")

# Compute the average price for each day of month
avg_price_day = df.groupby("day").agg({'adj_close': 'mean'}).reset_index()

# Plot
plot_df(avg_price_day, 'day', 'adj_close', 'day of month', 'average adjusted closing price (USD)', [], "no title",
        './figure/test_v6_day.pdf')
"""
data = [go.Scatter(
            x = avg_price_day['day'],
            y = avg_price_day['adj_close'],
            mode = 'markers',
            marker=dict(
                color='LightSkyBlue',
                size=15,
                line=dict(
                color='MediumPurple',
                width=2
                ))
        )]

layout = dict(xaxis = dict(title = 'day of month'),
              yaxis = dict(title = 'average adjusted closing price (USD)'))

fig = dict(data=data, layout=layout)
#py.iplot(fig, filename='StockPricePrediction_v6d_avg_price_dayofmonth')
#fig.write_image("./figure/test_v6_day.pdf")
"""

print("# In[1082]:")

# Compute the average price for each day of week
avg_price_dayofweek = df.groupby("dayofweek").agg({'adj_close': 'mean'}).reset_index()

# Plot

plot_df(avg_price_dayofweek, 'dayofweek', 'adj_close', 'day of week', 'average adjusted closing price (USD)', [],
        "no title", './figure/test_v6_day.pdf')
"""
data = [go.Scatter(
            x = avg_price_dayofweek['dayofweek'],
            y = avg_price_dayofweek['adj_close'],
            mode = 'markers',
            marker=dict(
                color='LightSkyBlue',
                size=15,
                line=dict(
                color='MediumPurple',
                width=2
                ))
        )]

layout = dict(xaxis = dict(title = 'day of week'),
              yaxis = dict(title = 'average adjusted closing price (USD)'))

fig = dict(data=data, layout=layout)
#py.iplot(fig, filename='StockPricePrediction_v6d_avg_price_dayofweek')
##fig.write_image("./figure/test_v6_day.pdf")
"""

print("# In[1083]:")

# Create lags
df_lags = add_lags(df, N, ['adj_close'])
df_lags

print("# In[1084]:")

# Compute correlation
features = [
    'adj_close',
    'year',
    'month',
    'week',
    'day',
    'dayofweek',
    'dayofyear',
    'is_month_end',
    'is_month_start',
    'is_quarter_end',
    'is_quarter_start',
    'is_year_end',
    'is_year_start'
]
for n in range(N, 0, -1):
    features.append("adj_close_lag_" + str(n))

corr_matrix = df_lags[features].corr()
corr_matrix["adj_close"].sort_values(ascending=False)

# Plot correlation for lag features only
features = ['adj_close']
for n in range(1, N + 1, 1):
    features.append("adj_close_lag_" + str(n))
corr_matrix = df_lags[features].corr()
z_list = []
for feat in features:
    z_list.append(corr_matrix.loc[:, feat][features])

print("# In[1086]:")

# Plot correlation for date features only
features = [
    'adj_close',
    'year',
    'month',
    'week',
    'day',
    'dayofweek',
    'dayofyear',
    'is_month_end',
    'is_month_start',
    'is_quarter_end',
    'is_quarter_start',
    'is_year_end',
    'is_year_start'
]

corr_matrix = df_lags[features].corr()

z_list = []
for feat in features:
    z_list.append(corr_matrix.loc[:, feat][features])

# # Split into train, validation, test

print("# In[1087]:")

lst_clmn_param = ["pred_day",
                  "N",
                  "H",
                  "RMSE",
                  "MAPE",
                  "MAE",
                  "ACCURACY",
                  "n_estimators",
                  "max_depth",
                  "learning_rate",
                  "min_child_weight",
                  "subsample",
                  "colsample_bytree",
                  "colsample_bylevel",
                  "gamma"]



df_lst_param = pd.DataFrame(columns= lst_clmn_param)

print("# In[1088]:")

for pred_day in pred_day_list:

    print("############################################################################")
    print("Predicting on day %d, date %s, with forecast horizon H = %d" % (pred_day, df.iloc[pred_day]['date'], H))
    train = df[pred_day - train_val_size:pred_day - val_size].copy()
    val = df[pred_day - val_size:pred_day].copy()
    train_val = df[pred_day - train_val_size:pred_day].copy()
    test = df[pred_day:pred_day + H].copy()
    print("train.shape = " + str(train.shape))
    print("val.shape = " + str(val.shape))
    print("train_val.shape = " + str(train_val.shape))
    print("test.shape = " + str(test.shape))

    # # Predict for a specific H (forecast horizon) and a specific date

    print("# In[1089]:")

    print("Get error metrics on validation set before hyperparameter tuning:")
    rmse_bef_tuning, mape_bef_tuning, mae_bef_tuning, accuracy_bef_tuning, preds_dict = get_error_metrics(train_val,
                                                                                                          train_size,
                                                                                                          N,
                                                                                                          H,
                                                                                                          seed=model_seed,
                                                                                                          n_estimators=n_estimators,
                                                                                                          max_depth=max_depth,
                                                                                                          learning_rate=learning_rate,
                                                                                                          min_child_weight=min_child_weight,
                                                                                                          subsample=subsample,
                                                                                                          colsample_bytree=colsample_bytree,
                                                                                                          colsample_bylevel=colsample_bylevel,
                                                                                                          gamma=gamma)
    print("before tuning RMSE = %0.3f" % rmse_bef_tuning)
    print("before tuning MAPE = %0.3f%%" % mape_bef_tuning)
    print("before tuning MAE = %0.3f%%" % mae_bef_tuning)
    print("before tuning ACCURACY = %0.3f%%" % accuracy_bef_tuning)

    print("# In[1090]:")

    best_rmse = rmse_bef_tuning
    best_mape = mape_bef_tuning
    best_mae = mae_bef_tuning
    best_accuracy = accuracy_bef_tuning

    row_param = set_param_row(pred_day, N, H, rmse_bef_tuning, mape_bef_tuning, mae_bef_tuning, accuracy_bef_tuning, n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, colsample_bylevel, gamma)
    df_lst_param = addRow(df_lst_param, row_param)

    print("# In[1091]:")

    print("Do prediction on test set:")
    test_rmse_bef_tuning, test_mape_bef_tuning, test_mae_bef_tuning, test_accuracy_bef_tuning, est, feature_importances, features = get_error_metrics_one_pred(
        df[pred_day - train_val_size:pred_day + H],
        train_size + val_size,
        N,
        H,
        seed=model_seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        gamma=gamma)

    print("test set RMSE = %0.3f" % test_rmse_bef_tuning)
    print("test set MAPE = %0.3f%%" % test_mape_bef_tuning)
    print("test set MAE = %0.3f" % test_mae_bef_tuning)
    print("test set ACCURACY = %0.3f" % test_accuracy_bef_tuning)

    save_row = False
    if (test_rmse_bef_tuning <= best_rmse):
        best_rmse = test_rmse_bef_tuning
        save_row = True
    if (test_mape_bef_tuning <= best_mape):
        best_mape = test_mape_bef_tuning
        save_row = True
    if (test_mae_bef_tuning <= best_mae):
        best_mae = test_mae_bef_tuning
        save_row = True
    if (test_accuracy_bef_tuning >= best_accuracy):
        best_accuracy = test_accuracy_bef_tuning
        save_row = True

    if(save_row == True):
        row_param = set_param_row(pred_day, N, H, test_rmse_bef_tuning, test_mape_bef_tuning, test_mae_bef_tuning, test_accuracy_bef_tuning,
                                  n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, colsample_bylevel, gamma)
        df_lst_param = addRow(df_lst_param, row_param)

    print("# In[1092]:")

    print("# In[1093]:")

    # View a list of the features and their importance scores
    imp = list(zip(features, feature_importances))
    imp.sort(key=lambda tup: tup[1], reverse=False)
    imp

    # # Predict for a specific H (forecast horizon) and a specific date, with hyperparam tuning

    print("# In[1094]:")

    # We use a constant for N here
    N_opt = N

    # ## Tuning n_estimators (default=100) and max_depth (default=3)

    print("# In[1095]:")

    error_rate = defaultdict(list)

    n_estimators_opt_param = []
    max_depth_opt_param = []


    param_label  = 'n_estimators'
    param2_label = 'max_depth'
    param3_label = 'learning_rate'
    param4_label = 'min_child_weight'
    param5_label = 'subsample'
    param6_label = 'gamma'
    param7_label = 'colsample_bytree'
    param8_label = 'colsample_bylevel'

    param_label = 'n_estimators'
    param_list = range(40, 61, 1)
    # param_list = range(30, 51, 1)
    # param_list = range(42, 56, 1)
    #param_list = [42, 44, 47]
    param2_label = 'max_depth'
    param2_list = [2, 3, 4, 5, 6, 7, 8, 9]
    #param2_list = [3, 5, 7, 9]

    cpt = 0
    tic = time.time()
    # for param in tqdm_notebook(param_list):
    titi = time.time()

    error_rate = defaultdict(list)

    for param in param_list:
        for param2 in param2_list:
            if (cpt % 10) == 0:
                print("cpt :", cpt)
                toto = time.time()
                print("Tuning loop x10 Minutes taken = {0:.2f}".format((toto - titi) / 60.0))
            cpt = cpt + 1
            rmse_mean, mape_mean, mae_mean, accuracy_mean, _ = get_error_metrics(train_val,
                                                                                 train_size,
                                                                                 N_opt,
                                                                                 H,
                                                                                 seed=model_seed,
                                                                                 n_estimators=param,
                                                                                 max_depth=param2,
                                                                                 learning_rate=learning_rate,
                                                                                 min_child_weight=min_child_weight,
                                                                                 subsample=subsample,
                                                                                 colsample_bytree=colsample_bytree,
                                                                                 colsample_bylevel=colsample_bylevel,
                                                                                 gamma=gamma)
            # Collect results
            error_rate[param_label].append(param)
            error_rate[param2_label].append(param2)
            error_rate[param3_label].append(learning_rate)
            error_rate[param4_label].append(min_child_weight)
            error_rate[param5_label].append(subsample)
            error_rate[param6_label].append(colsample_bytree)
            error_rate[param7_label].append(colsample_bytree)
            error_rate[param8_label].append(gamma)
            error_rate['rmse'].append(rmse_mean)
            error_rate['mape'].append(mape_mean)
            error_rate['mae'].append(mae_mean)
            error_rate['accuracy'].append(accuracy_mean)

    toto = time.time()
    print("Tuning loop Minutes taken = {0:.2f}".format((toto - titi) / 60.0))

    df_error_rate = pd.DataFrame(error_rate)
    df_error_rate.to_csv("./output/error_rate_step1_" + str(pred_day) + ".csv")

    print("Minutes taken = {0:.2f}".format((toto - titi) / 60.0))

    # Get optimum value for param and param2, using RMSE
    temp = df_error_rate[df_error_rate['rmse'] == df_error_rate['rmse'].min()]
    print("min RMSE = %0.3f" % df_error_rate['rmse'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])

    # Get optimum value for param and param2, using MAPE
    temp = df_error_rate[df_error_rate['mape'] == df_error_rate['mape'].min()]
    print("min MAPE = %0.3f%%" % df_error_rate['mape'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])

    # Get optimum value for param and param2, using MAE
    temp = df_error_rate[df_error_rate['mae'] == df_error_rate['mae'].min()]
    print("min MAE = %0.3f%%" % df_error_rate['mae'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])

    # Get optimum value for param and param2, using ACCURACY
    temp = df_error_rate[df_error_rate['accuracy'] == df_error_rate['accuracy'].max()]
    print("max ACCURACY = %0.3f%%" % df_error_rate['accuracy'].max())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])

    n_estimators_opt_param = list(set(n_estimators_opt_param))
    max_depth_opt_param = list(set(max_depth_opt_param))

    print("n_estimators_opt_param: ", n_estimators_opt_param)
    print("max_depth_opt_param: ", max_depth_opt_param)


    param_list = n_estimators_opt_param
    param2_list = max_depth_opt_param

    learning_rate_opt_param = []
    min_child_weight_opt_param = []
    param3_label = 'learning_rate'
    #param3_list = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]
    #param_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    param3_list = [0.001, 0.01, 0.1, 0.2, 0.4]
    # param_list = [0.2, 0.4]

    param4_label = 'min_child_weight'
    param4_list = range(5, 25, 1)
    #param4_list = [9, 11]

    cpt = 0
    titi = time.time()

    error_rate = defaultdict(list)

    # for param in tqdm_notebook(param_list):
    for param in param_list:
        for param2 in param2_list:
            for param3 in param3_list:
                for param4 in param4_list:
                    if (cpt % 10) == 0:
                        print("cpt :", cpt)
                    cpt = cpt + 1
                    rmse_mean, mape_mean, mae_mean, accuracy_mean, _ = get_error_metrics(train_val,
                                                                                         train_size,
                                                                                         N_opt,
                                                                                         H,
                                                                                         seed=model_seed,
                                                                                         n_estimators=param,
                                                                                         max_depth=param2,
                                                                                         learning_rate=param3,
                                                                                         min_child_weight=param4,
                                                                                         subsample=subsample,
                                                                                         colsample_bytree=colsample_bytree,
                                                                                         colsample_bylevel=colsample_bylevel,
                                                                                         gamma=gamma)
                    # Collect results
                    error_rate[param_label].append(param)
                    error_rate[param2_label].append(param2)
                    error_rate[param3_label].append(param3)
                    error_rate[param4_label].append(param4)
                    error_rate[param5_label].append(subsample)
                    error_rate[param6_label].append(colsample_bytree)
                    error_rate[param7_label].append(colsample_bytree)
                    error_rate[param8_label].append(gamma)
                    error_rate['rmse'].append(rmse_mean)
                    error_rate['mape'].append(mape_mean)
                    error_rate['mae'].append(mae_mean)
                    error_rate['accuracy'].append(accuracy_mean)

    toto = time.time()
    print("Tuning loop Minutes taken = {0:.2f}".format((toto - titi) / 60.0))

    df_error_rate = pd.DataFrame(error_rate)
    df_error_rate.to_csv("./output/error_rate2_" + str(pred_day) + ".csv")

    print("Minutes taken = {0:.2f}".format((toto - titi) / 60.0))

    # Get optimum value for param and param2, using RMSE
    temp = df_error_rate[df_error_rate['rmse'] == df_error_rate['rmse'].min()]
    print("min RMSE = %0.3f" % df_error_rate['rmse'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])

    # Get optimum value for param and param2, using MAPE
    temp = df_error_rate[df_error_rate['mape'] == df_error_rate['mape'].min()]
    print("min MAPE = %0.3f%%" % df_error_rate['mape'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])

    # Get optimum value for param and param2, using MAE
    temp = df_error_rate[df_error_rate['mae'] == df_error_rate['mae'].min()]
    print("min MAE = %0.3f%%" % df_error_rate['mae'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    n_estimators_opt_param.append(temp['learning_rate'].values[0])
    max_depth_opt_param.append(temp['min_child_weight'].values[0])

    # Get optimum value for param and param2, using ACCURACY
    temp = df_error_rate[df_error_rate['accuracy'] == df_error_rate['accuracy'].max()]
    print("max ACCURACY = %0.3f%%" % df_error_rate['accuracy'].max())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])

    n_estimators_opt_param = list(set(n_estimators_opt_param))
    max_depth_opt_param = list(set(max_depth_opt_param))
    learning_rate_opt_param = list(set(learning_rate_opt_param))
    min_child_weight_opt_param = list(set(min_child_weight_opt_param))

    print("n_estimators_opt_param: ", n_estimators_opt_param)
    print("max_depth_opt_param: ", max_depth_opt_param)
    print("learning_rate_opt_param: ", learning_rate_opt_param)
    print("min_child_weight_opt_param: ", min_child_weight_opt_param)

    param_list = n_estimators_opt_param
    param2_list = max_depth_opt_param
    param3_list = learning_rate_opt_param
    param4_list = min_child_weight_opt_param

    subsample_opt_param = []
    gamma_opt_param = []
    param5_label = 'subsample'
    param5_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    #param5_list = [0.1, 0.2, 0.3, 0.4]
    #param5_list = [0.1, 0.2, 0.4, 0.5, 0.6]
    #param_list = [0.1, 0.5]

    param6_label = 'gamma'
    #param6_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.3, 1.5]
    # param2_list = [0, 0.5, 1, 1.3, 1.5]
    param6_list = [0, 0.5, 1, 1.3, 1.5]

    cpt = 0
    titi = time.time()

    error_rate = defaultdict(list)

    # for param in tqdm_notebook(param_list):
    for param in param_list:
        for param2 in param2_list:
            for param3 in param3_list:
                for param4 in param4_list:
                    for param5 in param5_list:
                        for param6 in param6_list:
                            if(cpt % 10) == 0:
                                print("cpt :", cpt)
                            cpt = cpt + 1
                            rmse_mean, mape_mean, mae_mean, accuracy_mean, _ = get_error_metrics(train_val,
                                                                                                 train_size,
                                                                                                 N_opt,
                                                                                                 H,
                                                                                                 seed=model_seed,
                                                                                                 n_estimators=param,
                                                                                                 max_depth=param2,
                                                                                                 learning_rate=param3,
                                                                                                 min_child_weight=param4,
                                                                                                 subsample=param5,
                                                                                                 colsample_bytree=colsample_bytree,
                                                                                                 colsample_bylevel=colsample_bylevel,
                                                                                                 gamma=param6)
                            # Collect results
                            error_rate[param_label].append(param)
                            error_rate[param2_label].append(param2)
                            error_rate[param3_label].append(param3)
                            error_rate[param4_label].append(param4)
                            error_rate[param5_label].append(param5)
                            error_rate[param6_label].append(colsample_bytree)
                            error_rate[param7_label].append(colsample_bytree)
                            error_rate[param8_label].append(param6)
                            error_rate['rmse'].append(rmse_mean)
                            error_rate['mape'].append(mape_mean)
                            error_rate['mae'].append(mae_mean)
                            error_rate['accuracy'].append(accuracy_mean)

                            toto = time.time()
                            print("Tuning loop Minutes taken = {0:.2f}".format((toto - titi) / 60.0))


    df_error_rate = pd.DataFrame(error_rate)
    df_error_rate.to_csv("./output/error_rate3_" + str(pred_day) + ".csv")

    toc = time.time()
    print("Minutes taken = {0:.2f}".format((toto - titi) / 60.0))

    # Get optimum value for param and param2, using RMSE
    temp = df_error_rate[df_error_rate['rmse'] == df_error_rate['rmse'].min()]
    print("min RMSE = %0.3f" % df_error_rate['rmse'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])
    subsample_opt_param.append(temp['subsample'].values[0])
    gamma_opt_param.append(temp['gamma'].values[0])

    # Get optimum value for param and param2, using MAPE
    temp = df_error_rate[df_error_rate['mape'] == df_error_rate['mape'].min()]
    print("min MAPE = %0.3f%%" % df_error_rate['mape'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])
    subsample_opt_param.append(temp['subsample'].values[0])
    gamma_opt_param.append(temp['gamma'].values[0])

    # Get optimum value for param and param2, using MAPE
    temp = df_error_rate[df_error_rate['mae'] == df_error_rate['mae'].min()]
    print("min MAE = %0.3f%%" % df_error_rate['mae'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])
    subsample_opt_param.append(temp['subsample'].values[0])
    gamma_opt_param.append(temp['gamma'].values[0])

    # Get optimum value for param and param2, using ACCURACY
    temp = df_error_rate[df_error_rate['accuracy'] == df_error_rate['accuracy'].max()]
    print("max ACCURACY = %0.3f%%" % df_error_rate['accuracy'].max())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])
    subsample_opt_param.append(temp['subsample'].values[0])
    gamma_opt_param.append(temp['gamma'].values[0])

    n_estimators_opt_param = list(set(n_estimators_opt_param))
    max_depth_opt_param = list(set(max_depth_opt_param))
    learning_rate_opt_param = list(set(learning_rate_opt_param))
    min_child_weight_opt_param = list(set(min_child_weight_opt_param))
    subsample_opt_param = list(set(subsample_opt_param))
    gamma_opt_param = list(set(gamma_opt_param))

    print("n_estimators_opt_param: ", n_estimators_opt_param)
    print("max_depth_opt_param: ", max_depth_opt_param)
    print("learning_rate_opt_param: ", learning_rate_opt_param)
    print("min_child_weight_opt_param: ", min_child_weight_opt_param)
    print("subsample_opt_param: ", subsample_opt_param)
    print("gamma_opt_param: ", gamma_opt_param)

    param_list  = n_estimators_opt_param
    param2_list = max_depth_opt_param
    param3_list = learning_rate_opt_param
    param4_list = min_child_weight_opt_param
    param5_list = subsample_opt_param
    param6_list = gamma_opt_param


    colsample_bytree_opt_param = []
    colsample_bylevel_opt_param = []
    param7_label = 'colsample_bytree'
    #param7_list = [0.5, 0.9, 1]
    param7_list = [0.5, 0.8, 0.9, 1]
    #param7_list = [0.5, 1]

    param8_label = 'colsample_bylevel'
    #param8_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    param8_list = [0.5, 0.8, 1]

    error_rate = defaultdict(list)

    cpt = 0
    titi = time.time()
    # for param in tqdm_notebook(param_list):
    for param in param_list:
        for param2 in param2_list:
            for param3 in param3_list:
                for param4 in param4_list:
                    for param5 in param5_list:
                        for param6 in param6_list:
                            for param7 in param7_list:
                                for param8 in param8_list:
                                    if (cpt % 10) == 0:
                                        print("cpt :", cpt)
                                    cpt = cpt + 1
                                    rmse_mean, mape_mean, mae_mean, accuracy_mean, _ = get_error_metrics(train_val,
                                                                                                         train_size,
                                                                                                         N_opt,
                                                                                                         H,
                                                                                                         seed=model_seed,
                                                                                                         n_estimators=param,
                                                                                                         max_depth=param2,
                                                                                                         learning_rate=param3,
                                                                                                         min_child_weight=param4,
                                                                                                         subsample=param5,
                                                                                                         colsample_bytree=param7,
                                                                                                         colsample_bylevel=param8,
                                                                                                         gamma=param6)

                                    # Collect results
                                    error_rate[param_label].append(param)
                                    error_rate[param2_label].append(param2)
                                    error_rate[param3_label].append(param3)
                                    error_rate[param4_label].append(param4)
                                    error_rate[param5_label].append(param5)
                                    error_rate[param6_label].append(param6)
                                    error_rate[param7_label].append(param7)
                                    error_rate[param8_label].append(param8)
                                    error_rate['rmse'].append(rmse_mean)
                                    error_rate['mape'].append(mape_mean)
                                    error_rate['mae'].append(mae_mean)
                                    error_rate['accuracy'].append(accuracy_mean)

                                    toto = time.time()
                                    print("Tuning loop Minutes taken = {0:.2f}".format((toto - titi) / 60.0))

    df_error_rate = pd.DataFrame(error_rate)
    df_error_rate.to_csv("./output/error_rate4_" + str(pred_day) + ".csv")

    toto = time.time()
    print("Minutes taken = {0:.2f}".format((toto - titi) / 60.0))

    # Get optimum value for param and param2, using RMSE
    temp = df_error_rate[df_error_rate['rmse'] == df_error_rate['rmse'].min()]
    print("min RMSE = %0.3f" % df_error_rate['rmse'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])
    subsample_opt_param.append(temp['subsample'].values[0])
    gamma_opt_param.append(temp['gamma'].values[0])
    colsample_bytree_opt_param.append(temp['colsample_bytree'].values[0])
    colsample_bylevel_opt_param.append(temp['colsample_bylevel'].values[0])

    # Get optimum value for param and param2, using MAPE
    temp = df_error_rate[df_error_rate['mape'] == df_error_rate['mape'].min()]
    print("min MAPE = %0.3f%%" % df_error_rate['mape'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])
    subsample_opt_param.append(temp['subsample'].values[0])
    gamma_opt_param.append(temp['gamma'].values[0])
    colsample_bytree_opt_param.append(temp['colsample_bytree'].values[0])
    colsample_bylevel_opt_param.append(temp['colsample_bylevel'].values[0])

    # Get optimum value for param and param2, using MAPE
    temp = df_error_rate[df_error_rate['mae'] == df_error_rate['mae'].min()]
    print("min MAE = %0.3f%%" % df_error_rate['mae'].min())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])
    subsample_opt_param.append(temp['subsample'].values[0])
    gamma_opt_param.append(temp['gamma'].values[0])
    colsample_bytree_opt_param.append(temp['colsample_bytree'].values[0])
    colsample_bylevel_opt_param.append(temp['colsample_bylevel'].values[0])

    # Get optimum value for param and param2, using ACCURACY
    temp = df_error_rate[df_error_rate['accuracy'] == df_error_rate['accuracy'].max()]
    print("max ACCURACY = %0.3f%%" % df_error_rate['accuracy'].max())
    n_estimators_opt_param.append(temp['n_estimators'].values[0])
    max_depth_opt_param.append(temp['max_depth'].values[0])
    learning_rate_opt_param.append(temp['learning_rate'].values[0])
    min_child_weight_opt_param.append(temp['min_child_weight'].values[0])
    subsample_opt_param.append(temp['subsample'].values[0])
    gamma_opt_param.append(temp['gamma'].values[0])
    colsample_bytree_opt_param.append(temp['colsample_bytree'].values[0])
    colsample_bylevel_opt_param.append(temp['colsample_bylevel'].values[0])

    n_estimators_opt_param = list(set(n_estimators_opt_param))
    max_depth_opt_param = list(set(max_depth_opt_param))
    learning_rate_opt_param = list(set(learning_rate_opt_param))
    min_child_weight_opt_param = list(set(min_child_weight_opt_param))
    subsample_opt_param = list(set(subsample_opt_param))
    gamma_opt_param = list(set(gamma_opt_param))
    colsample_bytree_opt_param = list(set(colsample_bytree_opt_param))
    colsample_bylevel_opt_param = list(set(colsample_bylevel_opt_param))



    print("# In[1096]:")
    print("# In[1097]:")
    print("# In[1099]:")

    print("n_estimators_opt_param: ", n_estimators_opt_param)
    print("max_depth_opt_param: ", max_depth_opt_param)
    print("learning_rate_opt_param: ", learning_rate_opt_param)
    print("min_child_weight_opt_param: ", min_child_weight_opt_param)
    print("subsample_opt_param: ", subsample_opt_param)
    print("colsample_bytree_opt_param: ", colsample_bytree_opt_param)
    print("colsample_bylevel_opt_param: ", colsample_bylevel_opt_param)
    print("gamma_opt_param: ", gamma_opt_param)




    print("Final model")

    print("# In[1111]:")

    print("Get error metrics on validation set after hyperparameter tuning")

    """
    n_estimators_opt_param: [42]
    max_depth_opt_param: [3]
    learning_rate_opt_param: [0.4]
    min_child_weight_opt_param: [9]
    subsample_opt_param: [0.1]
    colsample_bytree_opt_param: [0.9]
    colsample_bylevel_opt_param: [1]
    gamma_opt_param: [1.3]

    print("Force selection:")
    print("n_estimators_opt_param: ", n_estimators_opt_param)
    print("max_depth_opt_param: ", max_depth_opt_param)
    print("learning_rate_opt_param: ", learning_rate_opt_param)
    print("min_child_weight_opt_param: ", min_child_weight_opt_param)
    print("subsample_opt_param: ", subsample_opt_param)
    print("colsample_bytree_opt_param: ", colsample_bytree_opt_param)
    print("colsample_bylevel_opt_param: ", colsample_bylevel_opt_param)
    print("gamma_opt_param: ", gamma_opt_param)
    """

    for n_estimators_opt in n_estimators_opt_param:
        for max_depth_opt in max_depth_opt_param:
            for learning_rate_opt in learning_rate_opt_param:
                for min_child_weight_opt in min_child_weight_opt_param:
                    for subsample_opt in subsample_opt_param:
                        for colsample_bytree_opt in colsample_bytree_opt_param:
                            for colsample_bylevel_opt in colsample_bylevel_opt_param:
                                for gamma_opt in gamma_opt_param:
                                    rmse_aft_tuning, mape_aft_tuning, mae_aft_tuning, accuracy_aft_tuning, preds_dict = get_error_metrics(
                                        train_val,
                                        train_size,
                                        N_opt,
                                        H,
                                        seed=model_seed,
                                        n_estimators=n_estimators_opt,
                                        max_depth=max_depth_opt,
                                        learning_rate=learning_rate_opt,
                                        min_child_weight=min_child_weight_opt,
                                        subsample=subsample_opt,
                                        colsample_bytree=colsample_bytree_opt,
                                        colsample_bylevel=colsample_bylevel_opt,
                                        gamma=gamma_opt)

                                    save_row = False
                                    if accuracy_aft_tuning >= best_accuracy:
                                        best_accuracy = accuracy_aft_tuning
                                        best_est = preds_dict
                                        save_row = True
                                    if rmse_aft_tuning <= best_rmse:
                                        best_rmse = rmse_aft_tuning
                                        save_row = True
                                    if mape_aft_tuning <= best_mape:
                                        best_mape = mape_aft_tuning
                                        save_row = True
                                    if mae_aft_tuning <= best_mae:
                                        best_mae = mae_aft_tuning
                                        save_row = True

                                    if (save_row == True):
                                        row_param = set_param_row(pred_day, N_opt, H, rmse_aft_tuning,
                                                                  mape_aft_tuning, mae_aft_tuning,
                                                                  accuracy_aft_tuning,
                                                                  n_estimators_opt, max_depth_opt, learning_rate_opt,
                                                                  min_child_weight_opt, subsample_opt, colsample_bytree_opt,
                                                                  colsample_bylevel_opt, gamma_opt)
                                        df_lst_param = addRow(df_lst_param, row_param)
                                        # Calculate RMSE
                                        print("tuning: RMSE on test set = %0.3f" % rmse_aft_tuning)
                                        # Calculate MAPE
                                        print("tuning: MAPE on test set = %0.3f%%" % mape_aft_tuning)
                                        # Calculate MAE
                                        print("tuning: MAE on test set = %0.3f%%" % mae_aft_tuning)
                                        # Calculate ACCURACY
                                        print("tuning: ACCURACY on test set = %0.3f%%" % accuracy_aft_tuning)
                                        best_n_estimators = n_estimators_opt
                                        best_max_depth = max_depth_opt
                                        best_learning_rate = learning_rate_opt
                                        best_min_child_weight = min_child_weight_opt
                                        best_subsample = subsample_opt
                                        best_colsample_bytree = colsample_bytree_opt
                                        best_colsample_bylevel = colsample_bylevel_opt
                                        best_gamma = gamma_opt

    print("after tuning RMSE = %0.3f" % best_rmse)
    print("after tuning MAPE = %0.3f%%" % best_mape)
    print("after tuning MAE = %0.3f" % best_mae)
    print("after tuning ACCURACY = %0.3f" % best_accuracy)

    df_lst_param.to_csv("./output/parameters_" + str(pred_day) + ".csv")

    preds_dict = best_est
    n_estimators_opt = best_n_estimators
    max_depth_opt = best_max_depth
    learning_rate_opt = best_learning_rate
    min_child_weight_opt = best_min_child_weight
    subsample_opt = best_subsample
    colsample_bytree_opt = best_colsample_bytree
    colsample_bylevel_opt = best_colsample_bylevel
    gamma_opt = best_gamma

    print("n_estimators_opt: ", best_n_estimators)
    print("max_depth_opt: ", best_max_depth)
    print("learning_rate_opt: ", best_learning_rate)
    print("min_child_weight_opt: ", best_min_child_weight)
    print("subsample_opt: ", best_subsample)
    print("colsample_bytree_opt: ", best_colsample_bytree)
    print("colsample_bylevel_opt: ", best_colsample_bylevel)
    print("gamma_opt: ", best_gamma)


    print("# In[1112]:")


    print("# In[1113]:")

    print("Set param:")
    #n_estimators_opt = 42
    #max_depth_opt = 3
    #learning_rate_opt = 0.4
    #min_child_weight_opt = 9
    #subsample_opt = 0.1
    #colsample_bytree_opt = 0.9
    #colsample_bylevel_opt = 1
    #gamma_opt = 1.3

    print("Do prediction on test set")
    test_rmse_aft_tuning, test_mape_aft_tuning, test_mae_aft_tuning, test_accuracy_bef_tuning, est, feature_importances, features = get_error_metrics_one_pred(
        df[pred_day - train_val_size:pred_day + H],
        train_size + val_size,
        N_opt,
        H,
        seed=model_seed,
        n_estimators=n_estimators_opt,
        max_depth=max_depth_opt,
        learning_rate=learning_rate_opt,
        min_child_weight=min_child_weight_opt,
        subsample=subsample_opt,
        colsample_bytree=colsample_bytree_opt,
        colsample_bylevel=colsample_bylevel_opt,
        gamma=gamma_opt)

    print("===> RMSE = %0.3f" % test_rmse_aft_tuning)
    print("===> MAPE = %0.3f%%" % test_mape_aft_tuning)
    print("===> MAE = %0.3f" % test_mae_aft_tuning)
    print("===> ACCURACY = %0.3f" % test_accuracy_bef_tuning)

    """
    print("# In[1113]: GS")
    print("Do prediction on test set with GS:")
    test_rmse_aft_tuning, test_mape_aft_tuning, test_mae_aft_tuning, test_accuracy_bef_tuning, est = get_error_metrics_GS(df[pred_day-train_val_size:pred_day+H],
                                                                                                                          train_size + val_size,
                                                                                                                          N_opt,
                                                                                                                          H,
                                                                                                                          seed=model_seed,
                                                                                                                          n_estimators=n_estimators_opt,
                                                                                                                          max_depth=max_depth_opt,
                                                                                                                          learning_rate=learning_rate_opt,
                                                                                                                          min_child_weight=min_child_weight_opt,
                                                                                                                          subsample=subsample_opt,
                                                                                                                          colsample_bytree=colsample_bytree_opt,
                                                                                                                          colsample_bylevel=colsample_bylevel_opt,
                                                                                                                          gamma=gamma_opt)

    print("RMSE = %0.3f" % test_rmse_bef_tuning)
    print("MAPE = %0.3f%%" % test_mape_bef_tuning)
    print("MAE = %0.3f" % test_mae_bef_tuning)
    print("ACCURACY = %0.3f" % test_accuracy_bef_tuning)
    """

    print("# In[1114]:")

    print("# In[1115]:")

    # View a list of the features and their importance scores
    imp = list(zip(features, feature_importances))
    imp.sort(key=lambda tup: tup[1], reverse=False)
    imp

    print("# In[1116]:")

    # ## Tuned params

    print("# In[1117]:")

    # Tuned params and before and after error metrics
    d = {'param': ['n_estimators', 'max_depth', 'learning_rate', 'min_child_weight', 'subsample', 'colsample_bytree',
                   'colsample_bylevel', 'gamma', 'val_rmse', 'val_mape', 'val_mae'],
         'before_tuning': [n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree,
                           colsample_bylevel, gamma, rmse_bef_tuning, mape_bef_tuning, mae_bef_tuning],
         'after_tuning': [n_estimators_opt, max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt,
                          colsample_bytree_opt, colsample_bylevel_opt, gamma_opt, rmse_aft_tuning, mape_aft_tuning,
                          mae_aft_tuning]}
    tuned_params = pd.DataFrame(d)
    tuned_params = tuned_params.round(3)
    tuned_params

    print("# In[1118]:")

    # Put tuned_params into pickle
    pickle.dump(tuned_params,
                open("./out/v6d_tuned_params_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + ".pickle", "wb"))

    tuned_params.to_csv("./out/v6d_tuned_params_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + ".pickle")

    print("# In[1119]:")

    # pickle.load(open("./out/v6d_tuned_params_" + "2017-03-06" + ".pickle", "rb"))

    print("# In[1120]:")

    toc1 = time.time()
    print("Total minutes taken = {0:.2f}".format((toc1 - tic1) / 60.0))

    # # Findings

    print("# In[1121]:")

    print("Predicting on day %d, date %s, with forecast horizon H = %d" % (
    pred_day, df.iloc[pred_day]['date'].strftime("%Y-%m-%d"), H))

    print("# In[1122]:")

    rmse_bef_tuning, rmse_aft_tuning

    print("# In[1123]:")

    test_rmse_bef_tuning, test_rmse_aft_tuning

    print("# In[1124]:")

    # Put results into pickle
    pickle.dump(rmse_bef_tuning,
                open("./out/v6d_val_rmse_bef_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + ".pickle",
                     "wb"))
    pickle.dump(rmse_aft_tuning,
                open("./out/v6d_val_rmse_aft_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + ".pickle",
                     "wb"))
    pickle.dump(test_rmse_bef_tuning,
                open("./out/v6d_test_rmse_bef_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + ".pickle",
                     "wb"))
    pickle.dump(test_mape_bef_tuning,
                open("./out/v6d_test_mape_bef_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + ".pickle",
                     "wb"))
    pickle.dump(test_mae_bef_tuning,
                open("./out/v6d_test_mae_bef_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + ".pickle",
                     "wb"))
    pickle.dump(test_rmse_aft_tuning,
                open("./out/v6d_test_rmse_aft_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + ".pickle",
                     "wb"))
    pickle.dump(test_mape_aft_tuning,
                open("./out/v6d_test_mape_aft_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + ".pickle",
                     "wb"))
    pickle.dump(test_mae_aft_tuning,
                open("./out/v6d_test_mae_aft_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + ".pickle",
                     "wb"))
    pickle.dump(est, open("./out/v6d_test_est_aft_tuning_" + df.iloc[pred_day]['date'].strftime("%Y-%m-%d") + ".pickle",
                          "wb"))

print("# In[1125]:")

df_lst_param.to_csv("./output/all_parameters.csv")

# Consolidate results
# H = 21                         # Forecast horizon, in days. Note there are about 252 trading days in a year
# train_size = 252*3             # Use 3 years of data as train set. Note there are about 252 trading days in a year
# val_size = 252                 # Use 1 year of data as validation set
# N = 10                          # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
results = defaultdict(list)
ests = {}  # the predictions
date_list = ['2017-01-03',
             '2017-03-06',
             '2017-05-04',
             '2017-07-05',
             '2017-09-01',
             '2017-11-01',
             '2018-01-03',
             '2018-03-06',
             '2018-05-04',
             '2018-07-05',
             '2018-09-04',
             '2018-11-01'
             ]
for date in date_list:
    results['date'].append(date)
    results['val_rmse_bef_tuning'].append(pickle.load(open("./out/v6d_val_rmse_bef_tuning_" + date + ".pickle", "rb")))
    results['val_rmse_aft_tuning'].append(pickle.load(open("./out/v6d_val_rmse_aft_tuning_" + date + ".pickle", "rb")))
    results['test_rmse_bef_tuning'].append(
        pickle.load(open("./out/v6d_test_rmse_bef_tuning_" + date + ".pickle", "rb")))
    results['test_rmse_aft_tuning'].append(
        pickle.load(open("./out/v6d_test_rmse_aft_tuning_" + date + ".pickle", "rb")))
    results['test_mape_bef_tuning'].append(
        pickle.load(open("./out/v6d_test_mape_bef_tuning_" + date + ".pickle", "rb")))
    results['test_mape_aft_tuning'].append(
        pickle.load(open("./out/v6d_test_mape_aft_tuning_" + date + ".pickle", "rb")))
    results['test_mae_bef_tuning'].append(pickle.load(open("./out/v6d_test_mae_bef_tuning_" + date + ".pickle", "rb")))
    results['test_mae_aft_tuning'].append(pickle.load(open("./out/v6d_test_mae_aft_tuning_" + date + ".pickle", "rb")))
    ests[date] = pickle.load(open("./out/v6d_test_est_aft_tuning_" + date + ".pickle", "rb"))

results = pd.DataFrame(results)
results

print("# In[1126]:")

# Generate a condensed dataframe of the above
results_short = defaultdict(list)
hyperparam_list = ['n_estimators',
                   'max_depth',
                   'learning_rate',
                   'min_child_weight',
                   #                    'subsample',
                   #                    'colsample_bytree',
                   #                    'colsample_bylevel',
                   #                    'gamma'
                   ]

for date in date_list:
    results_short['date'].append(date)
    results_short['RMSE'].append(pickle.load(open("./out/v6d_test_rmse_aft_tuning_" + date + ".pickle", "rb")))
    results_short['MAPE(%)'].append(pickle.load(open("./out/v6d_test_mape_aft_tuning_" + date + ".pickle", "rb")))
    results_short['MAE'].append(pickle.load(open("./out/v6d_test_mae_aft_tuning_" + date + ".pickle", "rb")))

    tuned_params = pickle.load(open("./out/v6d_tuned_params_" + date + ".pickle", "rb"))

    for hyperparam in hyperparam_list:
        results_short[hyperparam].append(tuned_params[tuned_params['param'] == hyperparam]['after_tuning'].values[0])

results_short = pd.DataFrame(results_short)
results_short

print("# In[1127]:")

results.mean()

print("# In[1128]:")

# Plot all predictions
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=df['date'],
                         y=df['adj_close'],
                         mode='lines',
                         name='adj_close',
                         line=dict(color='blue')))

# Plot the predictions
n = 0
for key in ests:
    i = df[df['date'] == key].index[0]
    fig.add_trace(go.Scatter(x=df[i:i + H]['date'],
                             y=ests[key],
                             mode='lines',
                             name='predictions',
                             line=dict(color=colors[n % len(colors)])))
    n = n + 1

fig.update_layout(yaxis=dict(title='USD'),
                  xaxis=dict(title='date'))
#fig.update_xaxes(range=['2017-01-03', '2018-12-28'])
#fig.update_yaxes(range=[110, 150])
# py.iplot(fig, filename='StockPricePrediction_v6d_xgboost_predictions')
fig.write_image("./figure/test_v6_xgboost_predictions_" + str(pred_day) + ".pdf",validate=False, engine="orca")

print("# In[1129]:")

# Plot scatter plot of actual values vs. predictions
fig = go.Figure()

n = 0
for key in ests:
    i = df[df['date'] == key].index[0]
    fig.add_trace(go.Scatter(x=df[i:i + H]['adj_close'],
                             y=ests[key],
                             mode='markers',
                             name='predictions',
                             line=dict(color=colors[n % len(colors)])))
    n = n + 1

fig.add_trace(go.Scatter(x=list(range(110, 155, 1)),
                         y=list(range(110, 155, 1)),
                         mode='lines',
                         name='actual values',
                         line=dict(color='blue')))

fig.update_layout(yaxis=dict(title='forecasts'),
                  xaxis=dict(title='adj_close'))
# py.iplot(fig, filename='StockPricePrediction_v6d_xgboost_actuals_vs_predictions')
fig.write_image("./figure/test_v6_xgboost_actuals_vs_predictions_" + str(pred_day) + ".pdf",validate=False, engine="orca")

print("# In[1131]:")

# Compare results with benchmark
all_results = pd.DataFrame({'Method': ['Last value', 'XGBoost w/o date features', 'XGBoost w date features'],
                            'RMSE': [2.53, 2.32, 2.42],
                            'MAPE(%)': [1.69, 1.53, 1.61],
                            'MAE': [2.26, 2.05, 2.15]})
all_results

print("# In[ ]:")




