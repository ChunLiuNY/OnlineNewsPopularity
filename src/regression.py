import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import random
from collections import defaultdict
import math

from train_test_data import *
from data_processing import *


def reg_model_eval(X_scaled, y, model):
  kf = KFold(n_splits=5, shuffle=True)
  rmse = []
  rsqr = []
  for train_idx, test_idx in kf.split(X_scaled):
    model.fit(X_scaled[train_idx], y.iloc[train_idx])
    y_pred_log = model.predict(X_scaled[test_idx])
    y_true_log = y.iloc[test_idx]
    y_pred = log_to_shares(y_pred_log)
    y_true = log_to_shares(y_true_log)
    rmse.append(mean_squared_error(y_true, y_pred,squared = False))
    rsqr.append(r2_score(y_true, y_pred))
  return rmse, rsqr

def reg_model_output(train_cols):
  df = get_data('../data/train.csv')
  df = get_log_shares(df)
  X, y = get_feature_target(df, train_cols)
  X_scaled = feature_scaling(X)
  return df, X_scaled, y

def print_reg_model_score(model, train_cols):
  df, X_scaled, y = reg_model_output(train_cols)
  rmse, rsqr = reg_model_eval(X_scaled, y, model)
  print("RMSE:", np.average(rmse))
  print("R^2:", np.var(rsqr))
