# The normal imports
import numpy as np # efficient numerical computations
import pandas as pd # data structures for data analysis
from numpy.random import randn

# These are the plotting modules adn libraries
import matplotlib as mpl # plotting (both interactive and to files)
import matplotlib.pyplot as plt
import seaborn as sns # extra plot types, elegant and readable plot style

# machine learning algorithms, dataset access
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import RANSACRegressor

sns.set_style('whitegrid')
# Command so that plots appear in the iPython Notebook
%matplotlib inline

# Data Load
boston = load_boston()
# put the data into data frame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target


# Select 13 attributes for the explanatory variables in 14 valuse
x = df.iloc[:, :-1].values
# MEDV(14th attribute) is for response value
y = df['MEDV'].values

# split data (70% training data, and 30% test data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Get Predicted value from Linear Regression
lr = LinearRegression()
# If you use RANSAC Regressor, take off this annotation
# lr = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, 
#                          residual_metric=lambda x: np.sum(np.abs(x), axis=1), 
#                          residual_threshold=5.0, random_state=0)
lr.fit(x_train, y_train)
y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

# Show Graph
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training Data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test Data')

plt.xlabel('Predict Value')
plt.ylabel('Residdual Value')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.title('[Multiple Variants - Price of House] Residual Analysis')
plt.legend(loc=2)
plt.show()

####################################
### Coefficient of Determination ###
####################################

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print('MSE-Trained Data: %.2f, Test Data: %.2f' %(mse_train, mse_test))

r2_train = r2_score(y_train, ytrain_pred)
r2_test = r2_score(y_test, y_test_pred)

print('MSE-Trained Data: %.2f, Test Data: %.2f' %(r2_train, r2_test))