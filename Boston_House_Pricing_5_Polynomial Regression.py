
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

sns.set_style('whitegrid')
# Command so that plots appear in the iPython Notebook
get_ipython().run_line_magic('matplotlib', 'inline') # %matplotlib inline

# Data Load
boston = load_boston()
# put the data into data frame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target


# In[42]:


# Set Training Data
x = df[['LSTAT']].values
y = df['MEDV'].values

lr = LinearRegression()


# In[43]:


quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)


# In[44]:


# Transforming to Polynomial dim2 and dim3
x_quad = quadratic.fit_transform(x)
x_cubic = cubic.fit_transform(x)


# In[45]:


# Simple Regression
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]
lr.fit(x, y)
y_lin_fit = lr.predict(x_fit)
l_r2 = r2_score(y, lr.predict(x))


# In[46]:


# Polynomial Regression (2 demension)
lr.fit(x_quad, y)
y_quad_fit = lr.predict(quadratic.fit_transform(x_fit))
q_r2 = r2_score(y, lr.predict(x_quad))


# In[47]:


# Polynomial Regression (3 demension)
lr.fit(x_cubic, y)
y_cubic_fit = lr.predict(cubic.fit_transform(x_fit))
c_r2 = r2_score(y, lr.predict(x_cubic))


# In[48]:


# Plotting
plt.scatter(x, y, label='traning data', c='lightgray')
plt.plot(x_fit, y_lin_fit, linestyle=':', label='linear fit(d=1), $R^2=%.2f$' %l_r2, c='blue', lw=3)
plt.plot(x_fit, y_quad_fit, linestyle='-', label='quad fit(d=2), $R^2=%.2f$' %q_r2, c='red', lw=3)
plt.plot(x_fit, y_cubic_fit, linestyle='--', label='cubic fit(d=2), $R^2=%.2f$' %c_r2, c='green', lw=3)
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend(loc=1)
plt.show()

