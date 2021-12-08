#  Sentiment analysis

It is a  Python program for sentiment analysis. Complete dataset is available [here](https://www.kaggle.com/kazanova/sentiment140). 
## About dataset
Original dataset contains 6 columns, but in this program we only keep 2 columns : tweet column and the sentiment labels(0 for negatif and 1 for positif). It is a binary classification problem. we have 800 000 tweets in each category. 
You can see some data visualisation in this program: word cloud, tweets length etc.

## Installing and requirements
You need to install :

 - Python
 - Pandas
 - Seaborn
 - Matplotlib
 - Sklearn
 - MLflow
  
## How does it work
To run this program you shoud do in a terminal or conda environment
```
sentiment_analysis.py
 ```
 for tracking the model with MLflow you can type this localhost in your browser:
 ```
 http://localhost:5000
 ```
 ## Evaluation
 ```
               precision    recall  f1-score   support

           0       0.80      0.76      0.78    160080
           1       0.77      0.80      0.79    159920

    accuracy                           0.78    320000
   macro avg       0.78      0.78      0.78    320000
weighted avg       0.78      0.78      0.78    320000
 ```
 
