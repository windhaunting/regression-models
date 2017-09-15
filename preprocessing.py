#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  14 10:49:40 2017

@author: fubao
"""
import pandas as pd
from numpy import log1p

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import FunctionTransformer


#pre processing data. dummy variable with categorical feature; scale feature; missing value etc
#use scikit-learn label and oneHotEncoder  -- method 1 
# --has bug  ValueError: setting an array element with a sequence when call this function
def dummyEncodeMethod1(df):
   # limit to categorical data using df.select_dtypes()
    X = df.select_dtypes(include=[object])
    #df.shape
    print ("X head: ", X.head(3))
    
    # encode labels with value between 0 and n_classes-1.
    le = LabelEncoder()
    # use df.apply() to apply le.fit_transform to all columns
    X_2 = X.apply(le.fit_transform)
    print ("X_2 head: ", X_2.head(3))

    #*** drop previous categorical columns
    #X.columns
    df.drop(X.columns, axis=1, inplace=True)

    #OneHotEncoder
    #Encode categorical integer features using a one-hot aka one-of-K scheme.
    enc = OneHotEncoder()
    onehotlabels = enc.fit_transform(X_2)
    
    dfX2 = pd.DataFrame(onehotlabels, index=range(0,onehotlabels.shape[0]), columns = range(0,onehotlabels.shape[1]), dtype=object)       #random index here
    
    df2 = pd.concat([df, dfX2], axis=1)        
    print ("onehotlabels.shape: ", onehotlabels.shape[1], df.shape, df2.shape, type(df2))
    return df2
    

#use pands get_dummies  -- method 2
def dummyEncodeMethod2(df):
    
   # limit to categorical data using df.select_dtypes()
    categoDf = df.select_dtypes(include=[object])
    #df.shape
    print ("categoDf head: ", categoDf.head(3))
    dfDummy = pd.get_dummies(categoDf)      #crete dummy variable or df factorize();    vs scikit-learn preprocessing Encoder

    #drop previous categorical columns
    df.drop(categoDf, axis=1, inplace=True) 

    df = pd.concat([dfDummy, df], axis=1)           #get purchase as the last column


    return df
    
#process missing value here
def preprocessNANMethod(df):
    #drop rows with all NaN
    df = df.dropna(axis=0, how='all', thresh=2)       #Keep only the rows with at least 2 non-na values:
    imputedArray = Imputer(missing_values="NaN", strategy='mean').fit_transform(df)
    
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
