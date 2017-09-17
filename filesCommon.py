#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 00:18:46 2017

@author: fubao
"""

'''
file related operation here
read input and write input operation
'''
import csv

import numpy as np
    
# Read in train and test data from files
def readTrainTestData(fileNameTrain, fileNameTrainLabel, fileTest):
	#print('Reading power plant dataset ...')
    
	train_x = np.loadtxt(fileNameTrain)
	train_y = np.loadtxt(fileNameTrainLabel)
	test_x = np.loadtxt(fileTest)

	return (train_x, train_y, test_x)



#write column wise file into file
def 	writeFileColumnwise(filePath, columnNameLst, columnsValues):
    
    writer = csv.writer(open(filePath, 'w'))
    writer.writerow(columnNameLst)
    writer.writerows(columnsValues)

