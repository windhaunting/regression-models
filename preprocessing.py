#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  14 10:49:40 2017

@author: fubao
"""
#import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import FunctionTransformer


'''
#check if there is None or NaN value process missing value here
def preprocessNANMethod(npArray):
    #drop rows with all NaN
    #print (np.equal(npArray, None))
    s = pd.DataFrame(npArray)
    print(s.isnull().values.any())
'''


#data transform, polynomial, log or exponential. etc.
def preprocessTransform(array):
    
    transArray = FunctionTransformer(log1p).fit_transform(array)
    return transArray
    
def preprocessScaler(array):
    #Transforms features by scaling each feature to a given range.
    # Standardize features by removing the mean and scaling to unit variance
    stanScalerArray = StandardScaler().fit_transform(array)
    #print("standard scaler: ", df.mean_)
    #df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

    #Transforms features by scaling each feature to a given range.
    rangeScalerArray = MinMaxScaler().fit_transform(stanScalerArray)
    
    return rangeScalerArray
