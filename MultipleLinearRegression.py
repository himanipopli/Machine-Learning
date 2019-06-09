# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 01:21:55 2019

@author: Himani Popli
"""

#Importing lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X=X[:,1:]

# Splitting dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Fitting Multiple Linear Regression Model on Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the test set results
Y_pred=regressor.predict(X_test)