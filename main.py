# This program predicts stock prices by using machine learning models.

import quandl
import numpy as np
from apiKey import apiKey
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

df = quandl.get("WIKI/FB", api_key=apiKey)

print(df.head())
