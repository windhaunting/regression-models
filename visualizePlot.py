#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  16 10:57:42 2017

@author: fubao
"""

import matplotlib.pyplot as plt


#visualize plot
#analyse and visualize data before training
def plotExploreDataPreTrain(trainX, trainY):
    

    # specifies the parameters of our graphs
    fig, axes = plt.subplots(nrows=trainX.shape[1]+1, ncols=1, figsize=(20, 20))
    fig.subplots_adjust(hspace=1.0) ## Create space between plots
    axes[0].hist(trainY, normed=True, bins=30)
    #axes[0,0].hist(trainY, normed=False, bins=30)
    print ("train X shape: ", trainX.shape[1])
    for i in range(1, trainX.shape[1] + 1):             #get every column
        
        axes[i].plot(trainX[:, i-1], trainY)
        
        plt.xlabel( str(i) + "th x feature")
        plt.ylabel("powerplant")
        plt.show()


#plot general figure common AfterTrain
def plotCommonAfterTrain(y_pred, y_true):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    
    
#analyse and visualize data after training; residualPlot
def plotResidualAfterTrain(y_pred, y_true):
    #plot residual plot
    plt.figure()
    plt.rcParams['agg.path.chunksize'] = 10000
    #print ("len y_pred, y_true: ", len(y_pred), len(y_true))
    #plt.scatter(x_test, y_test,  color='black')
    plt.plot(y_pred, y_true-y_pred, color='blue', linewidth=3)  #
    plt.show()
    

#plot cv time for different model
def plotCVTime(xPara, yTime, xLabel, yLabel, title, plotSavePdfPath):
    f = plt.figure()
    plt.plot(xPara, yTime, marker='o', linestyle='-', color='black')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    f.savefig(plotSavePdfPath, bbox_inches='tight')
