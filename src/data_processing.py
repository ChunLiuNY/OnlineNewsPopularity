import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
import random
import math

from train_test_data import *


def get_binary_label(df):
  df['label'] = df['shares'].map(lambda x: 1 if x >1400 else 0)
  return df

def class_split(df, lower_bound, upper_bound):
  df1 = df[(df['shares'] >= lower_bound) & (df['shares'] < upper_bound)]
  df2 = df[~((df['shares'] >= lower_bound) & (df['shares'] < upper_bound))]
  df2 = df2.sample(len(df1))
  return pd.concat([df1, df2])

def get_label(df, lower_bound, upper_bound):
  df['label'] = df['shares'].map(lambda x: 1 if (x >=lower_bound) & (x <upper_bound) else 0)
  return df

def get_log_shares(df):
  df['value'] = df['shares'].map(lambda x: math.log(x+1))
  return df

def get_feature_target(df, train_cols):
  X = df[train_cols]
  y = df['label']
  return X, y

def get_feature_target_reg(df, train_cols):
  X = df[train_cols]
  y = df['value']
  return X, y

def feature_scaling(X):
  scaler = MinMaxScaler()
  X_scaled = scaler.fit_transform(X)
  return X_scaled

def log_to_shares(log):
  return np.array([math.exp(x) - 1 for x in log])

def model_output(train_cols):
  df = get_data('../data/train.csv')
  df = get_binary_label(df)
  X, y = get_feature_target(df, train_cols)
  X_scaled = feature_scaling(X)
  return df, X_scaled, y

def model_output_multi(train_cols, lower_bound, upper_bound):
  df = get_data('../data/train.csv')
  df = class_split(df, lower_bound, upper_bound)
  df = get_label(df, lower_bound, upper_bound)
  X, y = get_feature_target(df, train_cols)
  X_scaled = feature_scaling(X)
  return df, X_scaled, y

def model_output_reg(train_cols):
  df = get_data('../data/train.csv')
  df = get_log_shares(df)
  X, y = get_feature_target(df, train_cols)
  X_scaled = feature_scaling(X)
  return df, X_scaled, y
