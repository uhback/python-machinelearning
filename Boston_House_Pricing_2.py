# Linear Regression

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

sns.set_style('whitegrid')
# Command so that plots appear in the iPython Notebook
%matplotlib inline

# Data Load
boston = load_boston()
# put the data into data frame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

# Linear Regression Function (model = LinearRegression)
def lin_regplot(x, y, model):
    plt.scatter(x, y, c='b')
    plt.plot(x, model.predict(x), c='r')    

x = df[['RM']].values
y = df[['MEDV']].values

slr = LinearRegression()
slr.fit(x,y)

slope = slr.coef_[0]
intercept = slr.intercept_

# Regression slope: 9.102 / Intercept: -34.671
print('Regression slope: %.3f\nIntercept: %.3f' %(slope, intercept))

# Draw graph
lin_regplot(x, y, slr)
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.title('the progress of boston house price in 1987[RM - MEDV]')
plt.show()