#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  14 10:49:40 2017

@author: fubao
"""
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import FunctionTransformer


#process missing value here
def preprocessNANMethod(npArray):
    #drop rows with all NaN
    npArray = npArray.dropna(axis=0, how='all', thresh=2)       #Keep only the rows with at least 2 non-na values:
    imputedArray = Imputer(missing_values="NaN", strategy='mean').fit_transform(npArray)
    
    #df = pd.DataFrame(imputedArray, index=df.index, columns=df.columns)
    #fill na
    
    return imputedArray

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
