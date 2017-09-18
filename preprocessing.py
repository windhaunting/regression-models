#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  14 10:49:40 2017

@author: fubao
"""
import pandas as pd
import numpy as np
from numpy import log1p

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import FunctionTransformer



#check if there is None or NaN value process missing value here
def preprocessNANMethod(array):
    #drop rows with all NaN
    #print (np.equal(npArray, None))
    s = pd.DataFrame(array)
    if (0 == s.isnull().values.any().sum()):
        return array
    else:
        #fill missing value
        s.fillna(s.mean(), inplace=True)
        return s.values 

#Transforms features by scaling each feature to a given range.
def preprocessNormalize(array):
    #Transforms features by scaling each feature to a given range.
    return MinMaxScaler().fit_transform(array)
    

#data transform, polynomial, log or exponential. etc.
def preprocessTransform(array):
    
    transArray = FunctionTransformer(log1p).fit_transform(array)
    return transArray
    

# Standardize features by removing the mean and scaling to unit variance
def preprocessStandardScaler(array):
    return StandardScaler().fit_transform(array)

