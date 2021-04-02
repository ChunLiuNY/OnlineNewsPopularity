import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_data(path):
  df = pd.read_csv(path)
  df = df.iloc[:,2:]
  return df

def col_clean(df):
  df = df.rename(columns = lambda x: x.strip())
  return df

def get_train_test(df):
  df = get_data('../data/OnlineNewsPopularity.csv')
  df = col_clean(df)
  train, test = train_test_split(df, test_size=0.2)
  train.to_csv(r'../data/train.csv',header=True)
  test.to_csv(r'../data/test.csv',header=True)



