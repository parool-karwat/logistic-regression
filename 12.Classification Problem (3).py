# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:50:17 2020

@author: Bhavin
"""

#Your First Proper Machine Learning Project

"""
Steps to finish this ML problem

1. Define Problem.
2. Prepare Data.
3. Evaluate Algorithms.
4. Improve Results.
5. Present Results.

The project is the classification of iris flowers.

List of all seaborn datasets is available here -
https://seaborn.pydata.org/generated/seaborn.load_dataset.html

"""

#imort libraries and check versions
import pandas as pd
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy as np
print('numpy: {}'.format(np.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
#seaborn
import seaborn as sns
print('seaborn: {}'.format(sns.__version__))
import numpy as np

df = sns.load_dataset('iris')
df
"""
Information about iris dataset and the machine learning problem is available at 
http://archive.ics.uci.edu/ml/datasets/iris
#Use logistic regression and tell me the accuracy
"""
df.shape
df.info()
df.head()
df.describe()
df.species.describe()
df.groupby('species').size() #This balance is important to check
df.groupby('species').describe().T

"""
We are going to look at two types of plots:

Univariate plots to better understand each attribute.
Multivariate plots to better understand the relationships between attributes.
"""
#Univariate
import matplotlib.pyplot as plt
# box and whisker plots
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

df.plot(kind='box')
plt.show()

df.plot(kind='box', subplots=True, layout=(2,2), sharex=True, sharey=True)
plt.show()

# histograms
df.hist()
plt.show()

df.plot(kind='hist', bins=25, subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#Multivariate Plots

# scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(10,8))
plt.show()

#Checking relations
df.groupby('species').get_group('setosa').boxplot(figsize=(8,6))
plt.show()

df.groupby('species').get_group('versicolor').boxplot(figsize=(8,6))
plt.show()

df.groupby('species').get_group('virginica').boxplot(figsize=(8,6))
plt.show()

colors = {'setosa':'b', 'versicolor':'r','virginica':'y'}

plt.figure(figsize=(12,10))
sns.pairplot(df, hue='species', palette = colors)

plt.figure(figsize=(12,10))
colors = {'setosa':'b', 'versicolor':'r','virginica':'y'}
sns.scatterplot('sepal_length','sepal_width', hue='species', data=df, palette = colors)

plt.figure(figsize=(12,10))
colors = {'setosa':'b', 'versicolor':'r','virginica':'y'}
sns.scatterplot('petal_length','petal_width', hue='species', data=df, palette = colors)

plt.figure(figsize=(12,10))
colors = {'setosa':'b', 'versicolor':'r','virginica':'y'}
sns.scatterplot('sepal_length','petal_length', hue='species', data=df, palette = colors)

plt.figure(figsize=(12,10))
colors = {'setosa':'b', 'versicolor':'r','virginica':'y'}
sns.scatterplot('sepal_width','petal_width', hue='species', data=df, palette = colors)


#Splitting the data

df.index.values

#Every third element is selected as test

test = df.iloc[::3,:]
test.head()

train = df.drop(df.iloc[::3,:].index, axis=0)
train.head(10)


#Using sample method
import pandas as pd
test = df.sample(n=50, random_state=10)
test.head()

train = df.drop(test.index, axis=0)
train.head()



#The industry standard

data = df.values
type(data) #always use this format

X = data[:,:-1]
X

y = data[:,-1]
y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#from logisti regression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#from kneighbors
from sklearn.neighbors import KNeighborsClassifier

kclassifier = KNeighborsClassifier()
kclassifier.fit(X_train, y_train)

accuracy = round(kclassifier.score(X_test, y_test) * 100, 2)
print(accuracy)
y_pred = kclassifier.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


