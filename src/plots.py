import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, log_loss
import random
from matplotlib import pyplot as plt


from train_test_data import *
from data_processing import *
from classification import *
from regression import *

def plot_target_features(axs, df, cols):
  for i, ax in enumerate(axs.flatten()):
    ax.scatter(df[cols[i]], df['shares'])
    ax.set_xlabel(cols[i])
    ax.set_ylabel('Number of Shares')

#mean decrease accuracy
def plot_feature_importance(X, y, train_cols):
  scores = defaultdict(list)
  rf = RandomForestClassifier()
  splitter = ShuffleSplit(10, test_size=.3)
  for train_idx, test_idx in splitter.split(X, y):
    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
    y_train, y_test = y[train_idx], y[test_idx]
    rf.fit(X_train, y_train)
    acc = f1_score(y_test, rf.predict(X_test))
    for i in range(X.shape[1]):
      X_t = X_test.copy()
      np.random.shuffle(X_t.iloc[:, i].values)
      shuff_acc = f1_score(y_test, rf.predict(X_t))
      scores[train_cols[i]].append((acc-shuff_acc)/acc)
  score_series = pd.DataFrame(scores).mean()
  scores = pd.DataFrame({'Mean Decrease Accuracy' : score_series})
  scores.sort_values(by='Mean Decrease Accuracy').plot(kind='barh', figsize = (16,16))

def calculate_payout(cb_matrix, model, X, threshold):
  return (confusion_matrix(model, X, threshold) * cb_matrix).values.sum()

def plot_profit_curve(ax,cb_matrix, model, X):
  thresholds = np.arange(0.0, 1.0, 0.01)
  profits = []
  for threshold in thresholds:
    profits.append(calculate_payout(cb_matrix, model, X, threshold))
  ax.plot(thresholds, profits)
  ax.set_xlabel('thresholds')
  ax.set_ylabel('profits')
  ax.set_title('Profit Curve')

def gb_learning_rt(X, y, N_ESTIMATORS = 200, N_FOLDS = 5):
  learning_rates = [1, 0.5, 0.1, 0.025, 0.01]
  N_LEARNING_RATES = len(learning_rates)
  train_scores = np.zeros((N_FOLDS, N_LEARNING_RATES, N_ESTIMATORS))
  test_scores = np.zeros((N_FOLDS, N_LEARNING_RATES, N_ESTIMATORS))
  folds = cv.KFold(n_splits=N_FOLDS, shuffle=True, random_state=1)

  for k, (train_idxs, test_idxs) in enumerate(folds.split(X)):
    X_train, y_train = X[train_idxs, :], y[train_idxs]
    X_test, y_test = X[test_idxs, :], y[test_idxs]

    models = [GradientBoostingClassifier(n_estimators=N_ESTIMATORS,
                                        max_depth=3, learning_rate=lr, subsample=0.5,
                                        random_state=154)
              for lr in learning_rates]
    for model in models:
      model.fit(X_train, y_train)
    for i, model in enumerate(models):
      for j, y_pred in enumerate(model.staged_predict(X_train)):
        train_scores[k, i, j] = model.loss_(y_train, y_pred)
    for i, model in enumerate(models):
      for j, y_pred in enumerate(model.staged_predict(X_test)):
        test_scores[k, i, j] = model.loss_(y_test, y_pred)

  mean_train_scores = np.mean(train_scores, axis=0)
  mean_test_scores = np.mean(test_scores, axis=0)
  return mean_train_scores, mean_test_scores, learning_rates

def plot_learning_rt(axes, learning_rates, N_ESTIMATORS, mean_train_scores, mean_test_scores):
  for i, rate in enumerate(learning_rates):
    ## train scores
    axes[0].plot(np.arange(N_ESTIMATORS) + 1, mean_train_scores[i, :],label="Learning Rate = " + str(rate))
    ## test scores
    axes[1].plot(np.arange(N_ESTIMATORS) + 1, mean_test_scores[i, :],label="Learning Rate = " + str(rate))
  ## common format
  for ax in axes:
    ax.legend(loc = "lower left")
    ax.set_xlabel('Number of Boosting Stages', fontsize=12)
    ax.set_ylabel('Average Squared Error', fontsize=12)
  ## subplot titles
  axes[0].set_title("Training Data")
  axes[1].set_title("Testing Data")
  plt.ylim([1.25, 1.43])
  _ = plt.suptitle("Effect of Varying the Learning Rate", fontsize=14)

def plot_model_performance(ax, train_cols):
  lst = ['Random Forest', 'Logistics Regression', 'Gradient Boosting']
  x = ['not_popular','mediocre','popular','super_popular']
  not_popular = get_model_score(train_cols, 0, 944)
  mediocre = get_model_score(train_cols, 945, 1400)
  popular = get_model_score(train_cols, 1401, 2800)
  super_popular = get_model_score(train_cols, 2801, 843300)
  for i in lst:
    y = [not_popular[i], mediocre[i], popular[i], super_popular[i]]
    ax.plot(x, y, label=i)
  ax.legend(loc = "upper right", fontsize=10)
  plt.ylim([0.5, 0.75])
  ax.set_ylabel('Average F1 Score', fontsize=12)
  ax.set_title("Multi-Classes Model Performance", fontsize=14)

def add_day_of_week(df):
  df_week = df.loc[:, ['weekday_is_monday', 'weekday_is_tuesday','weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday','weekday_is_saturday', 'weekday_is_sunday']]
  df_week['day_of_week'] = df_week.idxmax(axis=1)
  #df_week['day_of_week'] = df_week['day_of_week'].map(lambda x: x[11:])
  df_week['day_of_week'] = df_week['day_of_week'].map(lambda x: 1 if x =='weekday_is_monday' else x)
  df_week['day_of_week'] = df_week['day_of_week'].map(lambda x: 2 if x =='weekday_is_tuesday' else x)
  df_week['day_of_week'] = df_week['day_of_week'].map(lambda x: 3 if x =='weekday_is_wednesday' else x)
  df_week['day_of_week'] = df_week['day_of_week'].map(lambda x: 4 if x =='weekday_is_thursday' else x)
  df_week['day_of_week'] = df_week['day_of_week'].map(lambda x: 5 if x =='weekday_is_friday' else x)
  df_week['day_of_week'] = df_week['day_of_week'].map(lambda x: 6 if x =='weekday_is_saturday' else x)
  df_week['day_of_week'] = df_week['day_of_week'].map(lambda x: 7 if x =='weekday_is_sunday' else x)
  df = pd.concat([df, df_week['day_of_week']], axis=1)
  return df

def add_topic(df):
  df_topic = df.loc[:, ['data_channel_is_lifestyle','data_channel_is_entertainment', 'data_channel_is_bus','data_channel_is_socmed', 'data_channel_is_tech','data_channel_is_world']]
  df_topic['topic'] = df_topic.idxmax(axis=1)
  df_topic['topic'] = df_topic['topic'].map(lambda x: x[16:])
  df_topic['topic'] = df_topic['topic'].map(lambda x: 'business' if x=='bus' else x)
  df_topic['topic'] = df_topic['topic'].map(lambda x: 'socia media' if x=='socmed' else x)
  df = pd.concat([df, df_topic['topic']], axis=1)
  return df

def plot_week_topic(axs):
  df = get_data('../data/OnlineNewsPopularity.csv')
  df = col_clean(df)
  df = add_day_of_week(df)
  df = add_topic(df)
  df1=df.groupby(['day_of_week']).sum()['shares'].to_frame().reset_index()
  df2=df.groupby(['topic']).sum()['shares'].to_frame().reset_index()
  axs[0].bar(df1.iloc[:,0], df1.iloc[:,1])
  axs[1].bar(df2.iloc[:,0], df2.iloc[:,1])
  axs[0].set_ylabel('Number of Shares', fontsize=12)
  axs[1].set_ylabel('Number of Shares', fontsize=12)
  axs[0].set_title("Weekday Bar Graph of 39644 instances")
  axs[1].set_title("Topics Bar Graph of 39644 instances")






