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

sns.set_style('whitegrid')
# Command so that plots appear in the iPython Notebook
%matplotlib inline

boston = load_boston()
print boston.DESCR

# put the data into data frame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target

# Check null values
pd.isnull(df).any()

# Heatmap
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

# Pair plot
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM']
sns.pairplot(df[cols], size=2.5)
plt.show()
sns.reset_orig() # return to the origin style
