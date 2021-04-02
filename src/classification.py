import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import random
from collections import defaultdict
from sklearn.dummy import DummyClassifier
import sklearn.model_selection as cv

from train_test_data import *
from data_processing import *


def model_eval(X_scaled, y, model):
  kf = KFold(n_splits=5, shuffle=True)
  score = []
  for train_idx, test_idx in kf.split(X_scaled):
    model.fit(X_scaled[train_idx], y.iloc[train_idx])
    y_pred = model.predict(X_scaled[test_idx])
    y_true = y.iloc[test_idx]
    score.append(f1_score(y_true, y_pred))
  return score

def print_model_score(model, train_cols):
  df, X_scaled, y = model_output(train_cols)
  score = model_eval(X_scaled, y, model)
  print("Average F1 Score:", np.average(score))
  print("F1 Score Variance:", np.var(score))

def get_model_score(train_cols, lower_bound, upper_bound):
  df, X_scaled, y = model_output_multi(train_cols, lower_bound, upper_bound)
  base_model = DummyClassifier(strategy="stratified")
  rf = RandomForestClassifier(n_estimators=200, max_depth=59)
  lr = LogisticRegression(solver='sag')
  gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.5,random_state=154)
  models = {'Baseline': base_model,
              'Random Forest': rf,
              'Logistics Regression': lr,
              'Gradient Boosting': gb }
  res = {}
  for k,v in models.items():
    score = model_eval(X_scaled, y, v)
    res[k] = np.average(score)
  return res

def rmf_grid_search(X_scaled, y):
  param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 15, 30, 59],
    'max_features': ['auto', 15, 35, 50]}
  rf = RandomForestClassifier()
  grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
  grid_search.fit(X_scaled, y)
  return grid_search.best_params_

def lr_predict_threshold(model, X, threshold=0.5):
  '''Return prediction of the fitted binary-classifier model model on X using
  the specifed `threshold`. NB: class 0 is the positive class'''
  return np.where(model.predict_proba(X)[:, 0] > threshold,
                    model.classes_[0],
                    model.classes_[1])

def confusion_matrix(model, X, y, threshold=0.5):
  cf = pd.crosstab(y, lr_predict_threshold(model, X, threshold))
  cf.index.name = 'actual'
  cf.columns.name = 'predicted'
  return cf

