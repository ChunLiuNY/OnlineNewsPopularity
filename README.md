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

Which features are important to make prediction? Feature importance was obtained based on mean decrease accuracy in random forest model.  

To compute the importance of each feature:
* When one tree is grown, use it to predict the test samples and record F1 score.
* Randomly sort the feature (so that it is no longer correlated with the outcome) and do the prediction again. Compute the new F1 score.
* Average the change in F1 score across all trees.

Some important features:
*  Average keyword (avg. shares)
*  is_weekend
*  Minimum number of shares of Mashable links
*  Mashable data channel

## Exploaroty Data Analysis

Preliminary exploratory data analysis was conducted by creating scatter matrix visualizations and histograms to visualize correlations between features. 

The distribution of number of shares is right skewed as there are some viral articles with significant high number of shares. 

![Screen Shot 2021-04-02 at 12 27 26 AM](https://user-images.githubusercontent.com/26207455/113380461-49cd8200-934a-11eb-90c4-03239479704a.png)


Negative correlation is seen in number of images vs number of shares, number of videos vs number of shares, number of words in content vs number of shares and number of links vs number of shares. 

![Screen Shot 2021-03-31 at 2 29 11 PM](https://user-images.githubusercontent.com/26207455/113379554-d4f94880-9347-11eb-8f5e-dd8a8ab6088f.png)

The dataset contains more articles that are published during the weekdays than the weekends. This could be due to i) not enough data was collected for the weekends or ii) Mashable tends to publish fewer articles during the weekends. Life style related articles has the highest shares than other type of articles. 

![Screen Shot 2021-04-01 at 7 17 57 PM](https://user-images.githubusercontent.com/26207455/113379587-e6425500-9347-11eb-9ae8-b81527cbf3a4.png)



## Supervised Learning Models

### Logistics Regression

### Random Forest Classifier

### Gradient Boosting Classifier

## Business Insights


