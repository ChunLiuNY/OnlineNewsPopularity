# Online News Popularity Prediction
Mashable is a digital media website founded in 2005. It has 28 million followers on social media and 7.5 million shares per month. The goal of this analysis is to predict the popularity of an online news article in social network. Such a tool will help publishers and editors in maximizing the popularity of their articles and sell advertisement. 

## Dataset Description

The dataset in this analysis is publically available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity). It contains information of 39,644 articles published by Mashable in a period of two years from 2013 to 2015. The dataset is originally acquired and pre-processed by K. Fernandes et al (more details in this [paper](https://link.springer.com/chapter/10.1007/978-3-319-23485-4_53)). There are 58 predictive features, mainly categorized as below.

![Screen Shot 2021-04-01 at 11 39 54 PM](https://user-images.githubusercontent.com/26207455/113377891-96618f00-9343-11eb-8f10-8188e94901fb.png)

### Data Pre-processing

To prepare the data for modeling, the full dataset is divided into a master train and test set. The test set will not be used until the best model has been chosen by training and testing on the master train datset. 

Data pre-processing steps are as follows:
* removed spaces in column names
* URL was omitted when loading the data
* assigned 0 or 1 values for the label
* normalized features using MinMaxScaling

### Feature Importance

## Exploaroty Data Analysis

## Supervised Learning Models

### Logistics Regression

### Random Forest Classifier

### Gradient Boosting Classifier

## Business Insights


