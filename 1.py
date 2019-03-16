# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 19:37:59 2019

@author: Medeea
"""

#Data Preprocessing

#Importing the libraries

#3 libraries

#mathematic tools

import numpy as np

#plot nice charts

import matplotlib.pyplot as plt

#import datasets

import pandas as pd


#importing the dataset

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 3].values

#Taking care of Missing Data

from sklearn.preprocessing import Imputer

#create an objects at the class

#MEAN->AVERAGE
imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)

# 1:3
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#categorial data: country(3 category: Fr,Gr,Sp) and purchased(2 category: yes, no)
#encode text in to numbers

#encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#labelencoder_X.fit_transform(X[:, 0])
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into the Training set and Test set

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
