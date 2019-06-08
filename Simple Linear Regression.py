# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:14:29 2019

@author: Himani Popli
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the test set results
Y_pred=regressor.predict(X_test)

# Visualising the results
plt.scatter(X_train,Y_train,color='red')
plt.scatter(X_test,Y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color='blue')
#plt.plot(X_test,Y_pred,color='green')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()