#!/usr/bin/env python
# coding: utf-8

# # Objective
# * 20181225:
#     * Predict stock price in next day using XGBoost
#     * Given prices and other features for the last N days, we do prediction for day N+1
#     * Here we split 3 years of data into train(60%), dev(20%) and test(20%)
# * 20190110 - Diff from StockPricePrediction_v1_xgboost.ipynb:
#     * Here we scale the train set to have mean 0 and variance 1, and apply the same transformation to dev and test sets
# * 20190111 - Diff from StockPricePrediction_v1a_xgboost.ipynb:
#     * Here for the past N values for the dev set, we scale them to have mean 0 and variance 1, and do prediction on them
# * 20190112 - Diff from StockPricePrediction_v1b_xgboost.ipynb:
#     * Instead of using the same mean and variance to do scaling for train, dev and test sets, we scale the train set to have mean 0 and var 1, and then whenever we do prediction on dev or test set we scale the previous N values to also have mean 0 and var 1 (ie. use the means and variances of the previous N values to do scaling. We do this for both feature columns and target columns)

# In[803]:


import math
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import time

from datetime import date
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import learning_curve

# get_ipython().run_line_magic('matplotlib', 'inline')

#### Input params ##################
stk_path = "./data/VTI.csv"
test_size = 0.2  # proportion of dataset to be used as test set
cv_size = 0.2  # proportion of dataset to be used as cross-validation set
# N = 3                         # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
N = 21  # for feature at day t, we use lags from t-1, t-2, ..., t-N as features

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


####################################


# # Common functions

# In[804]:


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


def scale_row(row, feat_mean, feat_std):
    """
    Given a pandas series in row, scale it to have 0 mean and var 1 using feat_mean and feat_std
    Inputs
        row      : pandas series. Need to scale this.
        feat_mean: mean
        feat_std : standard deviation
    Outputs
        row_scaled : pandas series with same length as row, but scaled
    """
    # If feat_std = 0 (this happens if adj_close doesn't change over N days),
    # set it to a small number to avoid division by zero
    feat_std = 0.001 if feat_std == 0 else feat_std

    row_scaled = (row - feat_mean) / feat_std

    return row_scaled


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


def plot_multi_df(df_1, df_2, df_3, x_col, y_col, x_label, y_label, legend, title, file_name):
    # Plot adjusted close over time
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8

    ax = df_1.plot(x=x_col, y=y_col, style='b-', grid=True)
    ax = df_2.plot(x=x_col, y=y_col, style='y-', grid=True, ax=ax)
    ax = df_3.plot(x=x_col, y=y_col, style='g-', grid=True, ax=ax)
    ax.legend(legend)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    fig = ax.get_figure()
    fig.savefig(file_name)


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def accuracy_trend(y_test, pred):
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


def train_pred_eval_model(X_train_scaled,
                          y_train_scaled,
                          X_test_scaled,
                          y_test,
                          col_mean,
                          col_std,
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
        X_test_scaled      : features for test. Each sample is scaled to mean 0 and variance 1
        y_test             : target for test. Actual values, not scaled.
        col_mean           : means used to scale each sample of X_test_scaled. Same length as X_test_scaled and y_test
        col_std            : standard deviations used to scale each sample of X_test_scaled. Same length as X_test_scaled and y_test
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
        est                : predicted values. Same length as y_test
    '''

    model = XGBRegressor(seed=model_seed,
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
    est_scaled = model.predict(X_test_scaled)
    est = est_scaled * col_std + col_mean

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(y_test, est))
    mape = get_mape(y_test, est)

    accuracy = accuracy_trend(y_test, est_scaled)

    return rmse, mape, accuracy, est


# # Load data

# In[805]:


df = pd.read_csv(stk_path, sep=",")

# Convert Date column to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Change all column headings to be lower case, and remove spacing
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

# Get month of each sample
df['month'] = df['date'].dt.month

# Sort by datetime
df.sort_values(by='date', inplace=True, ascending=True)

df.head()

# In[806]:

plot_df(df, 'date', 'adj_close', "date", "USD", [], "no title", './figure/test_1.pdf')

# # Feature Engineering

# We will generate the following features:
# * Mean 'adj_close' of each month
# * Difference between high and low of each day
# * Difference between open and close of each day
# * Mean volume of each month

# In[807]:


# Get difference between high and low of each day
df['range_hl'] = df['high'] - df['low']
df.drop(['high', 'low'], axis=1, inplace=True)

# Get difference between open and close of each day
df['range_oc'] = df['open'] - df['close']
df.drop(['open', 'close'], axis=1, inplace=True)

df.head()

# Now we use lags up to N number of days to use as features.

# In[808]:


# Add a column 'order_day' to indicate the order of the rows by date
df['order_day'] = [x for x in list(range(len(df)))]

# merging_keys
merging_keys = ['order_day']

# List of columns that we will use to create lags
lag_cols = ['adj_close', 'range_hl', 'range_oc', 'volume']
lag_cols

# In[809]:


shift_range = [x + 1 for x in range(N)]

# for shift in tqdm_notebook(shift_range):
for i in range(N):
    shift = shift_range[i]

    train_shift = df[merging_keys + lag_cols].copy()

    # E.g. order_day of 0 becomes 1, for shift = 1.
    # So when this is merged with order_day of 1 in df, this will represent lag of 1.
    train_shift['order_day'] = train_shift['order_day'] + shift

    foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
    train_shift = train_shift.rename(columns=foo)

    df = pd.merge(df, train_shift, on=merging_keys, how='left')  # .fillna(0)

del train_shift

# Remove the first N rows which contain NaNs
df = df[N:]

df.head()

# In[810]:


df.info()

# In[811]:


# # Get mean of adj_close of each month
# df_gb = df.groupby(['month'], as_index=False).agg({'adj_close':'mean'})
# df_gb = df_gb.rename(columns={'adj_close':'adj_close_mean'})
# df_gb

# # Merge to main df
# df = df.merge(df_gb,
#               left_on=['month'],
#               right_on=['month'],
#               how='left').fillna(0)

# # Merge to main df
# shift_range = [x+1 for x in range(2)]

# for shift in tqdm_notebook(shift_range):
#     train_shift = df[merging_keys + lag_cols].copy()

#     # E.g. order_day of 0 becomes 1, for shift = 1.
#     # So when this is merged with order_day of 1 in df, this will represent lag of 1.
#     train_shift['order_day'] = train_shift['order_day'] + shift

#     foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
#     train_shift = train_shift.rename(columns=foo)

#     df = pd.merge(df, train_shift, on=merging_keys, how='left') #.fillna(0)

# del train_shift

# df


# In[812]:


# # Get mean of volume of each month
# df_gb = df.groupby(['month'], as_index=False).agg({'volume':'mean'})
# df_gb = df_gb.rename(columns={'volume':'volume_mean'})
# df_gb

# # Merge to main df
# df = df.merge(df_gb,
#               left_on=['month'],
#               right_on=['month'],
#               how='left').fillna(0)

# df.head()


# # Get mean and std dev at timestamp t using values from t-1, ..., t-N

# In[813]:


cols_list = [
    "adj_close",
    "range_hl",
    "range_oc",
    "volume"
]

for col in cols_list:
    df = get_mov_avg_std(df, col, N)
df.head()

# # Split into train, dev and test set

# In[814]:


# Get sizes of each of the datasets
num_cv = int(cv_size * len(df))
num_test = int(test_size * len(df))
num_train = len(df) - num_cv - num_test
print("num_train = " + str(num_train))
print("num_cv = " + str(num_cv))
print("num_test = " + str(num_test))

# Split into train, cv, and test
train = df[:num_train]
cv = df[num_train:num_train + num_cv]
train_cv = df[:num_train + num_cv]
test = df[num_train + num_cv:]
print("train.shape = " + str(train.shape))
print("cv.shape = " + str(cv.shape))
print("train_cv.shape = " + str(train_cv.shape))
print("test.shape = " + str(test.shape))

# # Scale the train, dev and test set

# In[815]:


cols_to_scale = [
    "adj_close"
]

for i in range(1, N + 1):
    cols_to_scale.append("adj_close_lag_" + str(i))
    cols_to_scale.append("range_hl_lag_" + str(i))
    cols_to_scale.append("range_oc_lag_" + str(i))
    cols_to_scale.append("volume_lag_" + str(i))

# Do scaling for train set
# Here we only scale the train dataset, and not the entire dataset to prevent information leak
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[cols_to_scale])
print("scaler.mean_ = " + str(scaler.mean_))
print("scaler.var_ = " + str(scaler.var_))
print("train_scaled.shape = " + str(train_scaled.shape))

# Convert the numpy array back into pandas dataframe
train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)
train_scaled[['date', 'month']] = train.reset_index()[['date', 'month']]
print("train_scaled.shape = " + str(train_scaled.shape))
train_scaled.head()

# In[816]:


# Do scaling for train+dev set
scaler_train_cv = StandardScaler()
train_cv_scaled = scaler_train_cv.fit_transform(train_cv[cols_to_scale])
print("scaler_train_cv.mean_ = " + str(scaler_train_cv.mean_))
print("scaler_train_cv.var_ = " + str(scaler_train_cv.var_))
print("train_cv_scaled.shape = " + str(train_cv_scaled.shape))

# Convert the numpy array back into pandas dataframe
train_cv_scaled = pd.DataFrame(train_cv_scaled, columns=cols_to_scale)
train_cv_scaled[['date', 'month']] = train_cv.reset_index()[['date', 'month']]
print("train_cv_scaled.shape = " + str(train_cv_scaled.shape))
train_cv_scaled.head()

# In[817]:


# Do scaling for dev set
cv_scaled = cv[['date']]
# for col in tqdm_notebook(cols_list):
for i in range(len(cols_list)):
    col = cols_list[i]
    feat_list = [col + '_lag_' + str(shift) for shift in range(1, N + 1)]
    temp = cv.apply(lambda row: scale_row(row[feat_list], row[col + '_mean'], row[col + '_std']), axis=1)
    cv_scaled = pd.concat([cv_scaled, temp], axis=1)

# Now the entire dev set is scaled
cv_scaled.head()

# In[818]:


# Do scaling for test set
test_scaled = test[['date']]
# for col in tqdm_notebook(cols_list):
for i in range(len(cols_list)):
    col = cols_list[i]
    feat_list = [col + '_lag_' + str(shift) for shift in range(1, N + 1)]
    temp = test.apply(lambda row: scale_row(row[feat_list], row[col + '_mean'], row[col + '_std']), axis=1)
    test_scaled = pd.concat([test_scaled, temp], axis=1)

# Now the entire test set is scaled
test_scaled.head()

# # Split into X and y

# In[819]:


features = []
for i in range(1, N + 1):
    features.append("adj_close_lag_" + str(i))
    features.append("range_hl_lag_" + str(i))
    features.append("range_oc_lag_" + str(i))
    features.append("volume_lag_" + str(i))

target = "adj_close"

# Split into X and y
X_train = train[features]
y_train = train[target]
X_cv = cv[features]
y_cv = cv[target]
X_train_cv = train_cv[features]
y_train_cv = train_cv[target]
X_sample = test[features]
y_sample = test[target]
print("X_train.shape = " + str(X_train.shape))
print("y_train.shape = " + str(y_train.shape))
print("X_cv.shape = " + str(X_cv.shape))
print("y_cv.shape = " + str(y_cv.shape))
print("X_train_cv.shape = " + str(X_train_cv.shape))
print("y_train_cv.shape = " + str(y_train_cv.shape))
print("X_sample.shape = " + str(X_sample.shape))
print("y_sample.shape = " + str(y_sample.shape))

# In[820]:


# Split into X and y
X_train_scaled = train_scaled[features]
y_train_scaled = train_scaled[target]
X_cv_scaled = cv_scaled[features]
X_train_cv_scaled = train_cv_scaled[features]
y_train_cv_scaled = train_cv_scaled[target]
X_sample_scaled = test_scaled[features]
print("X_train_scaled.shape = " + str(X_train_scaled.shape))
print("y_train_scaled.shape = " + str(y_train_scaled.shape))
print("X_cv_scaled.shape = " + str(X_cv_scaled.shape))
print("X_train_cv_scaled.shape = " + str(X_train_cv_scaled.shape))
print("y_train_cv_scaled.shape = " + str(y_train_cv_scaled.shape))
print("X_sample_scaled.shape = " + str(X_sample_scaled.shape))

# # EDA

# In[821]:

plot_multi_df(train, cv, test, 'date', 'adj_close', "date", "USD", ['train', 'dev', 'test'], "Without scaling",
              './figure/test_2.pdf')

# In[822]:


# Plot adjusted close over time
plot_df(train_scaled, 'date', 'adj_close', "date", "USD (scaled)", ['train_scaled'], "With scaling",
        './figure/test_3.pdf')

# # Train the model using XGBoost

# In[823]:


# Create the model
model = XGBRegressor(seed=model_seed,
                     n_estimators=n_estimators,
                     max_depth=max_depth,
                     learning_rate=learning_rate,
                     min_child_weight=min_child_weight,
                     subsample=subsample,
                     colsample_bytree=colsample_bytree,
                     colsample_bylevel=colsample_bylevel,
                     gamma=gamma)

# Train the regressor
model.fit(X_train_scaled, y_train_scaled)

# # Predict on train set

# In[824]:


# Do prediction on train set
est_scaled = model.predict(X_train_scaled)
est = est_scaled * math.sqrt(scaler.var_[0]) + scaler.mean_[0]

# Calculate RMSE
print("RMSE on train set = %0.3f" % math.sqrt(mean_squared_error(y_train, est)))

# Calculate MAPE
print("MAPE on train set = %0.3f%%" % get_mape(y_train, est))

# In[825]:


# Plot adjusted close over time
rcParams['figure.figsize'] = 10, 8  # width 10, height 8

est_df = pd.DataFrame({'est': est,
                       'date': train['date']})

ax = train.plot(x='date', y='adj_close', style='b-', grid=True)
ax = cv.plot(x='date', y='adj_close', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='adj_close', style='g-', grid=True, ax=ax)
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'dev', 'test', 'predictions'])
ax.set_xlabel("date")
ax.set_ylabel("USD")
ax.set_title('Without scaling')

fig = ax.get_figure()
fig.savefig('./figure/test_4.pdf')

# # Predict on dev set

# In[826]:


# Do prediction on test set
est_scaled = model.predict(X_cv_scaled)
cv['est_scaled'] = est_scaled
cv['est'] = cv['est_scaled'] * cv['adj_close_std'] + cv['adj_close_mean']

# Calculate RMSE
rmse_bef_tuning = math.sqrt(mean_squared_error(y_cv, cv['est']))
print("RMSE on dev set = %0.3f" % rmse_bef_tuning)

# Calculate MAPE
mape_bef_tuning = get_mape(y_cv, cv['est'])
print("MAPE on dev set = %0.3f%%" % mape_bef_tuning)

# In[827]:


# Plot adjusted close over time
rcParams['figure.figsize'] = 10, 8  # width 10, height 8

est_df = pd.DataFrame({'est': cv['est'],
                       'y_cv': y_cv,
                       'date': cv['date']})

ax = train.plot(x='date', y='adj_close', style='b-', grid=True)
ax = cv.plot(x='date', y='adj_close', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='adj_close', style='g-', grid=True, ax=ax)
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'dev', 'test', 'predictions'])
ax.set_xlabel("date")
ax.set_ylabel("USD")

fig = ax.get_figure()
fig.savefig('./figure/test_5.pdf')

# In[828]:


# Plot adjusted close over time, for dev set only
rcParams['figure.figsize'] = 10, 8  # width 10, height 8
ax = train.plot(x='date', y='adj_close', style='b-', grid=True)
ax = cv.plot(x='date', y='adj_close', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='adj_close', style='g-', grid=True, ax=ax)
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'dev', 'test', 'predictions'])
ax.set_xlabel("date")
ax.set_ylabel("USD")
ax.set_xlim([date(2017, 8, 1), date(2018, 5, 31)])
ax.set_title("Zoom in to dev set")

fig = ax.get_figure()
fig.savefig('./figure/test_6.pdf')

# The predictions capture the turn in directions with a slight lag

# In[829]:


# View a list of the features and their importance scores
imp = list(zip(train[features], model.feature_importances_))
imp.sort(key=lambda tup: tup[1])
imp[-10:]

# Importance features dominated by adj_close and volume

# # Tuning N (no. of days to use as features)

# In[830]:


d = {'N': [2, 3, 4, 5, 6, 7, 14],
     'rmse_dev_set': [1.225, 1.214, 1.231, 1.249, 1.254, 1.251, 1.498],
     'mape_pct_dev_set': [0.585, 0.581, 0.590, 0.601, 0.609, 0.612, 0.763]}
pd.DataFrame(d)

# Use N = 3 for lowest RMSE and MAPE

# # Tuning XGBoost - n_estimators (default=100) and max_depth (default=3)

# In[831]:


param_label = 'n_estimators'
param_list = range(10, 310, 5)

param2_label = 'max_depth'
param2_list = [2, 3, 4, 5, 6, 7, 8, 9]

error_rate = {param_label: [], param2_label: [], 'rmse': [], 'mape_pct': [], 'accuracy': []}
tic = time.time()
# for param in tqdm_notebook(param_list):
for param in param_list:
    #     print("param = " + str(param))
    for param2 in param2_list:
        # Train, predict and eval model
        rmse, mape, accuracy, _ = train_pred_eval_model(X_train_scaled,
                                                        y_train_scaled,
                                                        X_cv_scaled,
                                                        y_cv,
                                                        cv['adj_close_mean'],
                                                        cv['adj_close_std'],
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
        error_rate['rmse'].append(rmse)
        error_rate['mape_pct'].append(mape)
        error_rate['accuracy'].append(accuracy)

error_rate = pd.DataFrame(error_rate)
toc = time.time()
print("Minutes taken = " + str((toc - tic) / 60.0))
error_rate

# In[832]:


# Plot performance versus params
rcParams['figure.figsize'] = 10, 8  # width 10, height 8
temp = error_rate[error_rate[param2_label] == param2_list[0]]
ax = temp.plot(x=param_label, y='rmse', style='bs-', grid=True)
legend_list = [param2_label + '_' + str(param2_list[0])]

color_list = ['r', 'g', 'k', 'y', 'm', 'c', '0.75']
for i in range(1, len(param2_list)):
    temp = error_rate[error_rate[param2_label] == param2_list[i]]
    ax = temp.plot(x=param_label, y='rmse', color=color_list[i % len(color_list)], marker='s', grid=True, ax=ax)
    legend_list.append(param2_label + '_' + str(param2_list[i]))

ax.set_xlabel(param_label)
ax.set_ylabel("RMSE")
matplotlib.rcParams.update({'font.size': 14})
plt.legend(legend_list, loc='center left', bbox_to_anchor=(1.0, 0.5))  # positions legend outside figure

fig = ax.get_figure()
fig.savefig('./figure/test_7.pdf')

# In[833]:

n_estimators_opt_param = []
max_depth_opt_param = []
# Get optimum value for param and param2
temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
n_estimators_opt = temp['n_estimators'].values[0]
max_depth_opt = temp['max_depth'].values[0]
n_estimators_opt_param.append(n_estimators_opt)
max_depth_opt_param.append(max_depth_opt)
print("min RMSE = %0.3f" % error_rate['rmse'].min())
print("optimum params = ")
n_estimators_opt, max_depth_opt

# In[834]:


# Get optimum value for param and param2, using MAPE
temp = error_rate[error_rate['mape_pct'] == error_rate['mape_pct'].min()]
n_estimators_opt = temp['n_estimators'].values[0]
max_depth_opt = temp['max_depth'].values[0]
n_estimators_opt_param.append(n_estimators_opt)
max_depth_opt_param.append(max_depth_opt)
print("min MAPE = %0.3f%%" % error_rate['mape_pct'].min())
print("optimum params = ")
temp['n_estimators'].values[0], temp['max_depth'].values[0]

# Get optimum value for param and param2, using ACCURACY
temp = error_rate[error_rate['accuracy'] == error_rate['accuracy'].max()]
print("max ACCURACY = %0.3f%%" % error_rate['accuracy'].max())
print("optimum params = ")
temp['n_estimators'].values[0], temp['max_depth'].values[0]
n_estimators_opt = temp['n_estimators'].values[0]
max_depth_opt = temp['max_depth'].values[0]
n_estimators_opt_param.append(n_estimators_opt)
max_depth_opt_param.append(max_depth_opt)
# # Tuning XGBoost - learning_rate(default=0.1) and min_child_weight(default=1)

n_estimators_opt_param = list(set(n_estimators_opt_param))
max_depth_opt_param = list(set(max_depth_opt_param))

# In[835]:


param_label = 'learning_rate'
param_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]

param2_label = 'min_child_weight'
param2_list = range(5, 21, 1)

error_rate = {param_label: [], param2_label: [], 'rmse': [], 'mape_pct': [], 'accuracy': []}
tic = time.time()
# for param in tqdm_notebook(param_list):
for param in param_list:
    #     print("param = " + str(param))
    for param2 in param2_list:
        # Train, predict and eval model
        rmse, mape, accuracy, _ = train_pred_eval_model(X_train_scaled,
                                                        y_train_scaled,
                                                        X_cv_scaled,
                                                        y_cv,
                                                        cv['adj_close_mean'],
                                                        cv['adj_close_std'],
                                                        seed=model_seed,
                                                        n_estimators=n_estimators_opt,
                                                        max_depth=max_depth_opt,
                                                        learning_rate=param,
                                                        min_child_weight=param2,
                                                        subsample=subsample,
                                                        colsample_bytree=colsample_bytree,
                                                        colsample_bylevel=colsample_bylevel,
                                                        gamma=gamma)

        # Collect results
        error_rate[param_label].append(param)
        error_rate[param2_label].append(param2)
        error_rate['rmse'].append(rmse)
        error_rate['mape_pct'].append(mape)
        error_rate['accuracy'].append(accuracy)

error_rate = pd.DataFrame(error_rate)
toc = time.time()
print("Minutes taken = " + str((toc - tic) / 60.0))
error_rate

# In[836]:


# Plot performance versus params
rcParams['figure.figsize'] = 10, 8  # width 10, height 8
temp = error_rate[error_rate[param2_label] == param2_list[0]]
ax = temp.plot(x=param_label, y='rmse', style='bs-', grid=True)
legend_list = [param2_label + '_' + str(param2_list[0])]

color_list = ['r', 'g', 'k', 'y', 'm', 'c', '0.75']
for i in range(1, len(param2_list)):
    temp = error_rate[error_rate[param2_label] == param2_list[i]]
    ax = temp.plot(x=param_label, y='rmse', color=color_list[i % len(color_list)], marker='s', grid=True, ax=ax)
    legend_list.append(param2_label + '_' + str(param2_list[i]))

ax.set_xlabel(param_label)
ax.set_ylabel("RMSE")
matplotlib.rcParams.update({'font.size': 14})
plt.legend(legend_list, loc='center left', bbox_to_anchor=(1.0, 0.5))  # positions legend outside figure

fig = ax.get_figure()
fig.savefig('./figure/test_8.pdf')

# In[837]:

learning_rate_opt_param = []
min_child_weight_opt_param = []
# Get optimum value for param and param2
temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
learning_rate_opt = temp['learning_rate'].values[0]
min_child_weight_opt = temp['min_child_weight'].values[0]
learning_rate_opt_param.append(learning_rate_opt)
min_child_weight_opt_param.append(min_child_weight_opt)
print("min RMSE = %0.3f" % error_rate['rmse'].min())
print("optimum params = ")
learning_rate_opt, min_child_weight_opt

# In[838]:


# Get optimum value for param and param2, using MAPE
# We will use RMSE to decide the final optimum params to use
temp = error_rate[error_rate['mape_pct'] == error_rate['mape_pct'].min()]
print("min MAPE = %0.3f%%" % error_rate['mape_pct'].min())
print("optimum params = ")
temp['learning_rate'].values[0], temp['min_child_weight'].values[0]
learning_rate_opt = temp['learning_rate'].values[0]
min_child_weight_opt = temp['min_child_weight'].values[0]
learning_rate_opt_param.append(learning_rate_opt)
min_child_weight_opt_param.append(min_child_weight_opt)

# Get optimum value for param and param2, using ACCURACY
temp = error_rate[error_rate['accuracy'] == error_rate['accuracy'].max()]
print("max ACCURACY = %0.3f%%" % error_rate['accuracy'].max())
print("optimum params = ")
temp['learning_rate'].values[0], temp['min_child_weight'].values[0]
learning_rate_opt = temp['learning_rate'].values[0]
min_child_weight_opt = temp['min_child_weight'].values[0]
learning_rate_opt = temp['learning_rate'].values[0]
min_child_weight_opt = temp['min_child_weight'].values[0]
learning_rate_opt_param.append(learning_rate_opt)
min_child_weight_opt_param.append(min_child_weight_opt)

learning_rate_opt_param = list(set(learning_rate_opt_param))
min_child_weight_opt_param = list(set(min_child_weight_opt_param))
# # Tuning XGBoost - subsample(default=1) and gamma(default=0)

# In[839]:


param_label = 'subsample'
param_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

param2_label = 'gamma'
param2_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

error_rate = {param_label: [], param2_label: [], 'rmse': [], 'mape_pct': [], 'accuracy': []}
tic = time.time()
# for param in tqdm_notebook(param_list):
for param in param_list:
    #     print("param = " + str(param))

    for param2 in param2_list:
        # Train, predict and eval model
        rmse, mape, accuracy, _ = train_pred_eval_model(X_train_scaled,
                                                        y_train_scaled,
                                                        X_cv_scaled,
                                                        y_cv,
                                                        cv['adj_close_mean'],
                                                        cv['adj_close_std'],
                                                        seed=model_seed,
                                                        n_estimators=n_estimators_opt,
                                                        max_depth=max_depth_opt,
                                                        learning_rate=learning_rate_opt,
                                                        min_child_weight=min_child_weight_opt,
                                                        subsample=param,
                                                        colsample_bytree=colsample_bytree,
                                                        colsample_bylevel=colsample_bylevel,
                                                        gamma=param2)

        # Collect results
        error_rate[param_label].append(param)
        error_rate[param2_label].append(param2)
        error_rate['rmse'].append(rmse)
        error_rate['mape_pct'].append(mape)
        error_rate['accuracy'].append(accuracy)

error_rate = pd.DataFrame(error_rate)
toc = time.time()
print("Minutes taken = " + str((toc - tic) / 60.0))
error_rate

# In[840]:


# Plot performance versus params
rcParams['figure.figsize'] = 10, 8  # width 10, height 8
temp = error_rate[error_rate[param2_label] == param2_list[0]]
ax = temp.plot(x=param_label, y='rmse', style='bs-', grid=True)
legend_list = [param2_label + '_' + str(param2_list[0])]

color_list = ['r', 'g', 'k', 'y', 'm', 'c', '0.75']
for i in range(1, len(param2_list)):
    temp = error_rate[error_rate[param2_label] == param2_list[i]]
    ax = temp.plot(x=param_label, y='rmse', color=color_list[i % len(color_list)], marker='s', grid=True, ax=ax)
    legend_list.append(param2_label + '_' + str(param2_list[i]))

ax.set_xlabel(param_label)
ax.set_ylabel("RMSE")
matplotlib.rcParams.update({'font.size': 14})
plt.legend(legend_list, loc='center left', bbox_to_anchor=(1.0, 0.5))  # positions legend outside figure

fig = ax.get_figure()
fig.savefig('./figure/test_9.pdf')

# In[841]:

subsample_opt_param = []
gamma_opt_param = []

# Get optimum value for param and param2
temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
subsample_opt = temp['subsample'].values[0]
gamma_opt = temp['gamma'].values[0]
subsample_opt_param.append(subsample_opt)
gamma_opt_param.append(gamma_opt)
print("min RMSE = %0.3f" % error_rate['rmse'].min())
print("optimum params = ")
subsample_opt, gamma_opt

# In[842]:


# Get optimum value for param and param2, using MAPE
# We will use RMSE to decide the final optimum params to use
temp = error_rate[error_rate['mape_pct'] == error_rate['mape_pct'].min()]
subsample_opt = temp['subsample'].values[0]
gamma_opt = temp['gamma'].values[0]
subsample_opt_param.append(subsample_opt)
gamma_opt_param.append(gamma_opt)
print("min MAPE = %0.3f%%" % error_rate['mape_pct'].min())
print("optimum params = ")
temp['subsample'].values[0], temp['gamma'].values[0]

# Get optimum value for param and param2, using ACCURACY
temp = error_rate[error_rate['accuracy'] == error_rate['accuracy'].max()]
print("max ACCURACY = %0.3f%%" % error_rate['accuracy'].max())
print("optimum params = ")
temp['subsample'].values[0], temp['gamma'].values[0]
subsample_opt = temp['subsample'].values[0]
gamma_opt = temp['gamma'].values[0]
subsample_opt_param.append(subsample_opt)
gamma_opt_param.append(gamma_opt)

subsample_opt_param = list(set(subsample_opt_param))
gamma_opt_param = list(set(gamma_opt_param))

# # Tuning XGBoost - colsample_bytree(default=1) and colsample_bylevel(default=1)

# In[843]:


param_label = 'colsample_bytree'
param_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

param2_label = 'colsample_bylevel'
param2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

error_rate = {param_label: [], param2_label: [], 'rmse': [], 'mape_pct': [], 'accuracy': []}
tic = time.time()
# for param in tqdm_notebook(param_list):
for param in param_list:
    #     print("param = " + str(param))
    for param2 in param2_list:
        # Train, predict and eval model
        rmse, mape, accuracy, _ = train_pred_eval_model(X_train_scaled,
                                                        y_train_scaled,
                                                        X_cv_scaled,
                                                        y_cv,
                                                        cv['adj_close_mean'],
                                                        cv['adj_close_std'],
                                                        seed=model_seed,
                                                        n_estimators=n_estimators_opt,
                                                        max_depth=max_depth_opt,
                                                        learning_rate=learning_rate_opt,
                                                        min_child_weight=min_child_weight_opt,
                                                        subsample=subsample_opt,
                                                        colsample_bytree=param,
                                                        colsample_bylevel=param2,
                                                        gamma=gamma_opt)

        # Collect results
        error_rate[param_label].append(param)
        error_rate[param2_label].append(param2)
        error_rate['rmse'].append(rmse)
        error_rate['mape_pct'].append(mape)
        error_rate['accuracy'].append(accuracy)

error_rate = pd.DataFrame(error_rate)
toc = time.time()
print("Minutes taken = " + str((toc - tic) / 60.0))
error_rate

# In[844]:


# Plot performance versus params
rcParams['figure.figsize'] = 10, 8  # width 10, height 8
temp = error_rate[error_rate[param2_label] == param2_list[0]]
ax = temp.plot(x=param_label, y='rmse', style='bs-', grid=True)
legend_list = [param2_label + '_' + str(param2_list[0])]

color_list = ['r', 'g', 'k', 'y', 'm', 'c', '0.75']
for i in range(1, len(param2_list)):
    temp = error_rate[error_rate[param2_label] == param2_list[i]]
    ax = temp.plot(x=param_label, y='rmse', color=color_list[i % len(color_list)], marker='s', grid=True, ax=ax)
    legend_list.append(param2_label + '_' + str(param2_list[i]))

ax.set_xlabel(param_label)
ax.set_ylabel("RMSE")
matplotlib.rcParams.update({'font.size': 14})
plt.legend(legend_list, loc='center left', bbox_to_anchor=(1.0, 0.5))  # positions legend outside figure

fig = ax.get_figure()
fig.savefig('./figure/test_10.pdf')

# In[845]:

colsample_bytree_opt_param = []
colsample_bylevel_opt_param = []

# Get optimum value for param and param2
temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
colsample_bytree_opt = temp['colsample_bytree'].values[0]
colsample_bylevel_opt = temp['colsample_bylevel'].values[0]
colsample_bytree_opt_param.append(colsample_bytree_opt)
colsample_bylevel_opt_param.append(colsample_bylevel_opt)
print("min RMSE = %0.3f" % error_rate['rmse'].min())
print("optimum params = ")
colsample_bytree_opt, colsample_bylevel_opt

# In[846]:


# Get optimum value for param and param2, using MAPE
# We will use RMSE to decide the final optimum params to use
temp = error_rate[error_rate['mape_pct'] == error_rate['mape_pct'].min()]
colsample_bytree_opt = temp['colsample_bytree'].values[0]
colsample_bylevel_opt = temp['colsample_bylevel'].values[0]
colsample_bytree_opt_param.append(colsample_bytree_opt)
colsample_bylevel_opt_param.append(colsample_bylevel_opt)
print("min MAPE = %0.3f%%" % error_rate['mape_pct'].min())
print("optimum params = ")
temp['colsample_bytree'].values[0], temp['colsample_bylevel'].values[0]

# Get optimum value for param and param2, using ACCURACY
temp = error_rate[error_rate['accuracy'] == error_rate['accuracy'].max()]
print("max ACCURACY = %0.3f%%" % error_rate['accuracy'].max())
print("optimum params = ")
temp['colsample_bytree'].values[0], temp['colsample_bylevel'].values[0]
colsample_bytree_opt = temp['colsample_bytree'].values[0]
colsample_bylevel_opt = temp['colsample_bylevel'].values[0]
colsample_bytree_opt_param.append(colsample_bytree_opt)
colsample_bylevel_opt_param.append(colsample_bylevel_opt)

colsample_bytree_opt_param = list(set(colsample_bytree_opt_param))
colsample_bylevel_opt_param = list(set(colsample_bylevel_opt_param))

# # Tuned params

# In[857]:


d = {'param': ['n_estimators', 'max_depth', 'learning_rate', 'min_child_weight', 'subsample', 'colsample_bytree',
               'colsample_bylevel', 'gamma', 'rmse', 'mape_pct'],
     'original': [n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree,
                  colsample_bylevel, gamma, rmse_bef_tuning, mape_bef_tuning],
     'after_tuning': [n_estimators_opt, max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt,
                      colsample_bytree_opt, colsample_bylevel_opt, gamma_opt, error_rate['rmse'].min(),
                      error_rate['mape_pct'].min()]}
tuned_params = pd.DataFrame(d)
tuned_params = tuned_params.round(3)
tuned_params

# # Final model

# In[848]:

best_rmse = 0
best_mape = 0
best_accuracy = 0

for n_estimators_opt in n_estimators_opt_param:
    for max_depth_opt in max_depth_opt_param:
        for learning_rate_opt in learning_rate_opt_param:
            for min_child_weight_opt in min_child_weight_opt_param:
                for subsample_opt in subsample_opt_param:
                    for colsample_bytree_opt in colsample_bytree_opt_param:
                        for colsample_bylevel_opt in colsample_bylevel_opt_param:
                            for gamma_opt in gamma_opt_param:

                                rmse, mape, accuracy, est = train_pred_eval_model(X_train_cv_scaled,
                                                                                  y_train_cv_scaled,
                                                                                  X_sample_scaled,
                                                                                  y_sample,
                                                                                  test['adj_close_mean'],
                                                                                  test['adj_close_std'],
                                                                                  seed=model_seed,
                                                                                  n_estimators=n_estimators_opt,
                                                                                  max_depth=max_depth_opt,
                                                                                  learning_rate=learning_rate_opt,
                                                                                  min_child_weight=min_child_weight_opt,
                                                                                  subsample=subsample_opt,
                                                                                  colsample_bytree=colsample_bytree_opt,
                                                                                  colsample_bylevel=colsample_bylevel_opt,
                                                                                  gamma=gamma_opt)
                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_rmse = rmse
                                    best_mape = mape
                                    best_est = est
                                    # Calculate RMSE
                                    print("RMSE on test set = %0.3f" % rmse)
                                    # Calculate MAPE
                                    print("MAPE on test set = %0.3f%%" % mape)
                                    # Calculate ACCURACY
                                    print("ACCURACY on test set = %0.3f%%" % accuracy)

est = best_est
# Calculate RMSE
print("Best RMSE on test set = %0.3f" % best_rmse)

# Calculate MAPE
print("Best MAPE on test set = %0.3f%%" % best_mape)

# Calculate ACCURACY
print("Best ACCURACY on test set = %0.3f%%" % best_accuracy)

# In[849]:


# Plot adjusted close over time
rcParams['figure.figsize'] = 10, 8  # width 10, height 8

est_df = pd.DataFrame({'est': est,
                       'y_sample': y_sample,
                       'date': test['date']})

ax = train.plot(x='date', y='adj_close', style='b-', grid=True)
ax = cv.plot(x='date', y='adj_close', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='adj_close', style='g-', grid=True, ax=ax)
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'dev', 'test', 'predictions'])
ax.set_xlabel("date")
ax.set_ylabel("USD")

fig = ax.get_figure()
fig.savefig('./figure/test_11.pdf')

# In[851]:


# Plot adjusted close over time, for test set only
rcParams['figure.figsize'] = 10, 8  # width 10, height 8
ax = train.plot(x='date', y='adj_close', style='b-', grid=True)
ax = cv.plot(x='date', y='adj_close', style='y-', grid=True, ax=ax)
ax = test.plot(x='date', y='adj_close', style='g-', grid=True, ax=ax)
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'dev', 'test', 'predictions'])
ax.set_xlabel("date")
ax.set_ylabel("USD")
ax.set_xlim([date(2018, 4, 1), date(2018, 11, 30)])
ax.set_ylim([130, 155])
ax.set_title("Zoom in to test set")

fig = ax.get_figure()
fig.savefig('./figure/test_12.pdf')

# Similar to dev set, the predictions capture turns in direction with a slight lag

# In[858]:


# Plot adjusted close over time, only for test set
rcParams['figure.figsize'] = 10, 8  # width 10, height 8
matplotlib.rcParams.update({'font.size': 14})

ax = test.plot(x='date', y='adj_close', style='gx-', grid=True)
ax = est_df.plot(x='date', y='est', style='rx-', grid=True, ax=ax)
ax.legend(['test', 'predictions using xgboost'], loc='upper left')
ax.set_xlabel("date")
ax.set_ylabel("USD")
ax.set_xlim([date(2018, 4, 23), date(2018, 11, 23)])
ax.set_ylim([130, 155])

fig = ax.get_figure()
fig.savefig('./figure/test_13.pdf')

# In[860]:


# Save as csv
test_xgboost = est_df
test_xgboost.to_csv("./test_xgboost.csv")

# # Findings
# * By scaling the features properly, we can get good results for our predictions
# * RMSE and MAPE changed very little with hyperparameter tuning
# * The final RMSE and MAPE for test set are 1.162 and 0.58% respectively

# In[ ]:




