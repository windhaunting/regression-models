#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  15 18:38:42 2017

@author: fubao
"""

#feature selection module
#http://scikit-learn.org/stable/modules/feature_selection.html
#show feature selection methods here
import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from itertools import combinations, chain
from sklearn.metrics.cluster import normalized_mutual_info_score

from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFE

#Filter use variance statistics to do feature selection, select ones bigger than bigger than threshold
def featureSelectionFilterVariance01(df, threshold):
    #filter method
    #use variance:
    varSelector = VarianceThreshold(threshold)        # threshold = 0.1;  select features variances bigger than threshold 
    
    varSelector.fit_transform(df)
    #idxs = varSelector.get_support(indices=True)
    #print("featureSelection01 varArray: ", idxs)
    
    df = df.iloc[:, varSelector.get_support(indices=False)]
    #return varArray, idxs
    print("featureSelectionVariance01 df: ", df.shape)
    return df


#Filter use linear correlation statistics to do feature selection;  suitable only for linear relationship
def featureSelectionFilterCorrelation02(df, threshold):
    #get column list
    # 1st calculate the feature pair; 2nd use X -> y;   df contains X and y
    print ("featureSelectionFilterCorrelation02 df column", len(df.columns))
    Y = df.iloc[:,-1]          #df['Purchase']   df.iloc[:,-1] not probably the purchase  ycolumn
    #dfX = df.iloc[:, :-1]      #create a view , not to delete;  df.drop(df.columns[[-1,]], axis=1, inplace=True)
    #df.drop(df.columns[[-1,]], axis=1, inplace=True) df.drop(df.index[2])
    
    df.drop(df.columns[-1], axis=1, inplace=True)    #df.drop(df['Purchase'], axis=1, inplace=True)
    
    correlations = {}
    columns = df.columns.tolist()
    
    for col_a, col_b in combinations(columns, 2):
        correlations[col_a + '__' + col_b] = pearsonr(df.loc[:, col_a], df.loc[:, col_b])

    dfCorr = pd.DataFrame.from_dict(correlations, orient='index')
    dfCorr.columns = ['PCC', 'p-value']
    
    print ("featureSelectionFilterCorrelation02 result1: ", dfCorr.shape)
    
    #select one of the features in the feature pair with the absolute PCC larger than threshold
    dfCorr = dfCorr[dfCorr['PCC'] <= threshold]
    print ("featureSelectionFilterCorrelation02 result2: ", dfCorr.shape)
    
    colLsts = [f.split("__") for f in dfCorr.index.tolist()]
    cols = list(set(chain(*colLsts)))
    print ("cols:    ", len(cols))
    df = pd.concat([df[cols], Y], axis=1)
    print("featureSelectionFilterCorrelation02 final df: ", df.shape, df.columns)   #df['Purchase'])
    
    return df

#use mutual information to do feature selection.
#calculate all feature pairs with normalized mutual information(NMI); too cost for big feature set
#calculate feature vs predict value for regression model, filter too low NMI value
def featureSelectionFilterMutualInfo03(df, threshold):
    print ("featureSelectionFilterMutualInfo03 df column", len(df.columns))
    Y = df.iloc[:,-1]          #df['Purchase']   df.iloc[:,-1] not probably the purchase  ycolumn
    
    df.drop(df.columns[-1], axis=1, inplace=True)    #df.drop(df['Purchase'], axis=1, inplace=True)
    
    correlations = {}
    columns = df.columns.tolist()
    
    for col_a, col_b in combinations(columns, 2):
        correlations[col_a + '__' + col_b] = normalized_mutual_info_score(df.loc[:, col_a], df.loc[:, col_b])

    dfMI = pd.DataFrame.from_dict(correlations, orient='index')
    dfMI.columns = ['Normalized_Mutual_InfoScore']
    print ("featureSelectionFilterMutualInfo03 result2: ", dfMI)
    dfMI = dfMI[dfMI['Normalized_Mutual_InfoScore'] <= threshold]
    
    colLsts = [f.split("__") for f in dfMI.index.tolist()]
    cols = list(set(chain(*colLsts)))
    print ("cols:    ", len(cols))
    df = pd.concat([df[cols], Y], axis=1)
    print("featureSelectionFilterMutualInfo03 final df: ", df.shape, df.columns)   #df['Purchase'])
    
    return df

#wrapper select kbest using a function, such as chi2, f_regression to calculate x vs y
def featureSelectionWrapperKBest(df, k):
    X = df.drop(df.columns[-1], axis=1, inplace=False)            # inplace=True)
    Y = df.iloc[:,-1]      #Y = df['Purchase']                 #slicing create 2D list;
    print ('featureSelectionWrapperKBest: ', X.shape, Y.shape, )
    
    XSelector= SelectKBest(f_regression, k)            #chi2 error with unlabeled type
    XSelector.fit_transform(X.values, Y.as_matrix())
    
    #df = pd.concat([df[cols], Y], axis=1)
    dfXNew = df.iloc[:, XSelector.get_support(indices=False)]
    df = pd.concat([dfXNew, Y], axis=1)
    print ('featureSelectionWrapperKBest final: ', df.shape)
    return df

#based on RFE method; embedded
def featureSelectionEmbeddedRFE(df, threshold):
    x = 1
    