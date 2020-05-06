# This program predicts stock prices by using machine learning models.

import quandl
import numpy as np
from apiKey import apiKey
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Take a look at the data
df = quandl.get("WIKI/FB", api_key=apiKey)
# print(df.head())

# Get the adjusted lose price
df = df[['Adj. Close']]

# A variable for predicting 'n' days into the future
forecast_out = 1

# Create another column (the target or dependent variable) shifted 'n' units up
df['Prediction'] = df[['Adj. Close']].shift(-1)
print(df.head())
