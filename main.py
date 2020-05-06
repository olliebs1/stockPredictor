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
forecast_out = 30

# Create another column (the target or dependent variable) shifted 'n' units up
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
# print(df.tail())

# Create the independant data set (x)
# Convert the dataframe to a numpy array
X = np.array(df.drop(['Prediction'], 1))

# Remove the last 'n' rows
X = X[:-forecast_out]

# Create the dependant data set (y)
# Convert the dataframe to a numpy array (all values including the NaN's)
y = np.array(df['Prediction'])

# Get all of the y values except the last 'n' rows
y = y[:-forecast_out]

# Split the data into 80% training and 20% testing

train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
