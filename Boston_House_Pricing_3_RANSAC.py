
# coding: utf-8

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
from sklearn.linear_model import RANSACRegressor

sns.set_style('whitegrid')
# Command so that plots appear in the iPython Notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# Data Load
boston = load_boston()
# put the data into data frame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

x = df[['RM']].values
y = df[['MEDV']].values

# apply RANSAC
ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, 
                         residual_metric=lambda x: np.sum(np.abs(x), axis=1), 
                         residual_threshold=5.0, random_state=0)

ransac.fit(x,y)


inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_x = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_x[:, np.newaxis])
plt.scatter(x[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
plt.scatter(x[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')

plt.plot(line_x, line_y_ransac, c = 'red')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.title('the progress of boston house price in 1987[RM - MEDV]')
plt.show()

slope = ransac.estimator_.coef_[0]
intercept = ransac.estimator_.intercept_
print('Regression slope: %.3f\nIntercept: %.3f' %(slope, intercept))

