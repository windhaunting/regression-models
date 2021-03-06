# Import python modules
import numpy as np
import time
from kaggle import kaggleize

from filesCommon import readTrainTestData

from preprocessing import preprocessNANMethod
from preprocessing import preprocessNormalize
from preprocessing import preprocessTransform
from preprocessing import preprocessStandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

from visualizePlot import plotExploreDataPreTrain
from visualizePlot import plotCommonAfterTrain
from visualizePlot import plotResidualAfterTrain
from visualizePlot import plotCVTime
class clsregressionHw(object):
 
    def __init__(self):
      pass

    
    # read in train and test data of powerPlant
    def readDataPowerPlant(self):
        
        #transferNPtoDataframe():
        print('Reading power plant dataset ...')
        fileNameTrain = '../../Data/PowerOutput/data_train.txt'
        fileNameTrainLabel = '../../Data/PowerOutput/labels_train.txt'
        fileTest = '../../Data/PowerOutput/data_test.txt'
        trainX, trainY, testX = readTrainTestData(fileNameTrain, fileNameTrainLabel, fileTest)
                
        print (" power data shape trainX, trainY, testX: ", trainX.shape, trainY.shape, testX.shape)

        return (trainX, trainY, testX)
    

    # read in train and test data of indoor locationzation
    def read_data_localization_indoors(self):
        print('Reading indoor localization dataset ...')
        fileNameTrain = '../../Data/IndoorLocalization/data_train.txt'
        fileNameTrainLabel = '../../Data/IndoorLocalization/labels_train.txt'
        fileTest = '../../Data/IndoorLocalization/data_test.txt'
        trainX, trainY, testX = readTrainTestData(fileNameTrain, fileNameTrainLabel, fileTest)
                
        print (" IndoorLocalization data shape trainX, trainY, testX: ", trainX.shape, trainY.shape, testX.shape)

        return (trainX, trainY, testX)
    
   # Compute MAE
    def computeMAEError(self, y_hat, y):
        	# mean absolute error
        return np.abs(y_hat - y).mean()

    # select model parameter by using CV
    def modelSelectionCV(self, trainX, trainY, kfold, modelFunc, *args):

        kf = KFold(n_splits=kfold)
        averageMAE = 0.0
        sumMAE = 0.0
        for trainIndex, testIndex in kf.split(trainX):
            #print("TRAIN:", trainIndex, "TEST:", testIndex)
            xSplitTrain, XSplitTest = trainX[trainIndex], trainX[testIndex]
            ySplitTrain, ySplitTest = trainY[trainIndex], trainY[testIndex]
            
            #neigh = KNeighborsRegressor(n_neighbors=nNeighbor)
            model =  modelFunc(*args)
            model.fit(xSplitTrain, ySplitTrain)
            
            #print ("parameter: ", neigh.get_params(deep=True))
            predYSplitTest = model.predict(XSplitTest)
            
            #plot here
            #plotCommonAfterTrain(predYSplitTest, ySplitTest)
            #plotResidualAfterTrain(predYSplitTest, ySplitTest)
            
            #print ("predYSplitTest : ", predYSplitTest)
            mAE =  self.computeMAEError(predYSplitTest, ySplitTest)
            #print ("cv MAE error: ",i, mAE)
            sumMAE += mAE

        averageMAE  = sumMAE/kfold
        return averageMAE
    
    
    # use whole train data to do train and then test
    def trainTestWholeData(self, trainX, trainY, testX, modelFunc, *args):
        model =  modelFunc(*args)
        model.fit(trainX, trainY)
            
        #print ("parameter: ", neigh.get_params(deep=True))
        predY = model.predict(testX)
        
        return predY
    
    
    #execute model selection of KNN for power plant or indoor localization
    def executeTrainKNN(self, data, kfold, knnNeighbors, fileTestOutputKNN):
        trainX = data[0]
        trainY = data[1]
        testX = data[2]
        #print ("train X: ", trainX.shape)

        #use  k nearest neighbor knn
        i = 0
        smallestMAE = 2^32
        bestNNeighbor = knnNeighbors[0]
        
        for nNeighbor in knnNeighbors:
            args = (nNeighbor, 'uniform', 'auto', 30, 2,'minkowski', None, 8)
            averageMAE = self.modelSelectionCV(trainX, trainY, kfold, KNeighborsRegressor, *args)
            i += 1
            print ("averageMAE cv MAE error KNN: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestMAE = averageMAE
                bestNNeighbor = nNeighbor
        
        print (" bestNNeighbor KNN: ", smallestMAE, kfold, bestNNeighbor)
        predY = self.trainTestWholeData(trainX, trainY, testX, KNeighborsRegressor, bestNNeighbor)
        #print ("predY : KNN", predY)
        #output to file
        kaggleize(predY, fileTestOutputKNN)

        return (smallestMAE, kfold, bestNNeighbor)
    
    #execute linear regression powerPlant or indoor localization    
    def executeTrainLR(self, data, kfold, alphaLst, fileTestOutputLRRidge, fileTestOutputLRLasso):
        trainX = data[0]
        trainY = data[1]
        testX = data[2]

        #trainX = preprocessNormalize(trainX)           #not working
        #trainX = preprocessStandardScaler(trainX)      #might work
        
        #ridge begins
        smallestMAE = 2^32
        bestAlpha = alphaLst[0]
    
        for alpha in alphaLst:
            averageMAE = self.modelSelectionCV(trainX, trainY, kfold, Ridge, alpha)
            print ("averageMAE cv MAE error Ridge: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestMAE = averageMAE
                bestAlpha = alpha
        
        print (" bestAlpha Ridge: ", bestAlpha)
        predY = self.trainTestWholeData(trainX, trainY, testX, Ridge, bestAlpha)
        #print ("predY Ridge: ", predY)
        #output to file
        kaggleize(predY, fileTestOutputLRRidge)
        
        #lasso begins
        smallestMAE = 2^32
        bestAlpha = alphaLst[0]
        
        for alpha in alphaLst:
            averageMAE = self.modelSelectionCV(trainX, trainY, kfold, Lasso, alpha)
            print ("averageMAE cv MAE error Lasso: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestMAE = averageMAE
                bestAlpha = alpha
        
        print (" bestAlpha Lasso: ", bestAlpha)
        predY = self.trainTestWholeData(trainX, trainY, testX, Lasso, bestAlpha)
        #print ("predY Lasso: ", predY)
        #output to file
        kaggleize(predY, fileTestOutputLRLasso)
        
    
    #execute Decision tree powerPlant or indoor localization      
    def executeTrainDT(self, data, kfold, depthLst, fileTestOutputDT):
        trainX = data[0]
        trainY = data[1]
        testX = data[2]
        
        smallestMAE = 2^32
        bestDepth = depthLst[0]
        
        paraLstX = depthLst
        timeLstY = []

        for depth in depthLst:
            timeBegin = time.time()             #time begin
            
            args = ("mse", "best", depth)         #mae takes long long time?   # {"criterion": "mae", "splitter": "best", "max_depth": depth} 
            averageMAE = self.modelSelectionCV(trainX, trainY, kfold, DecisionTreeRegressor, *args)

            print ("averageMAE cv MAE error DT: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestMAE = averageMAE
                bestDepth = depth
            
            timeEnd = time.time()          #time end
            timeLstY.append((timeEnd-timeBegin)* 1000)
        #plot cv time
        fileNamePart = fileTestOutputDT.split("/")[2]
        title = "Time of cv for " + fileNamePart
        plotCVTime(paraLstX, timeLstY, "Decision tree depth", "Time (in Millisecond)", title, "../Figures/DTCVTime" + fileNamePart + ".pdf")
        
        args = ("mse", "best", bestDepth)            # {"criterion": "mae", "splitter": "best", "max_depth": bestDepth} 
        print (" bestDepth DT: ",smallestMAE,  kfold,  bestDepth)
        predY = self.trainTestWholeData(trainX, trainY, testX, DecisionTreeRegressor, *args)
        #print ("predY DT: ", predY)
        #output to file
        if fileTestOutputDT != "":
            kaggleize(predY, fileTestOutputDT)
        

        return (smallestMAE, kfold, bestDepth)

     #for assignment questions:
    def predictDifferentModels(self):
        
        #read power plant data
        dataPowerPlant = self.readDataPowerPlant()
    
        # Decision tree begins
        print (" -----begin decision tree for power plant--------")
        depthLst = [3, 6, 9, 12, 15]              #range(1, 20) try different alpha from test
        fileTestOutputDT  = "../Predictions/PowerOutput/best_DT.csv"
        kfold = 5
        self.executeTrainDT(dataPowerPlant, kfold, depthLst, fileTestOutputDT)
            
        #knn begins
        print (" -----begin knn regression for power plant--------")
        knnNeighbors = [3,5,10,20,25]   #range(1, 30)    #len(trainX), 2)              
        fileTestOutputKNN  = "../Predictions/PowerOutput/best_knn.csv"
        kfold = 5
        self.executeTrainKNN(dataPowerPlant, kfold, knnNeighbors, fileTestOutputKNN)
        
        #linear regression begins
        print (" -----begin linear regression ridge and lasso for power plant--------")
        alphaLst = [1e-6, 1e-4, 1e-2, 1, 10]              #try different alpha from test
        fileTestOutputLRRidge  = "../Predictions/PowerOutput/best_lr_ridge.csv"    
        fileTestOutputLRLasso  = "../Predictions/PowerOutput/best_lr_lasso.csv"    
        kfold = 5
        self.executeTrainLR(dataPowerPlant, kfold, alphaLst, fileTestOutputLRRidge, fileTestOutputLRLasso)
        

        # predict for indoor localization here
        dataIndoor = self.read_data_localization_indoors()
        
        # Decision tree begins
        print (" -----begin decision tree for indoor localization--------")
        depthLst = [20,25,30,35,40]            #range(1, 20) try different alpha from test
        fileTestOutputDT  = "../Predictions/IndoorLocalization/best_DT.csv"
        kfold = 5
        self.executeTrainDT(dataIndoor, kfold, depthLst, fileTestOutputDT)
        
        #knn begins
        print (" -----begin knn for indoor localization--------")
        knnNeighbors = [3,5,10,20,25]   #range(1, 30)    #len(trainX), 2) 
        fileTestOutputKNN  = "../Predictions/IndoorLocalization/best_knn.csv"
        kfold = 5
        self.executeTrainKNN(dataIndoor, kfold, knnNeighbors, fileTestOutputKNN)
        
        #linear regression begins
        print (" -----begin linear regression of ridge and lasso for indoor localization--------")
        alphaLst = [1e-6, 1e-4, 1e-2, 1, 10]              #try different alpha from test
        fileTestOutputLRRidge  = "../Predictions/IndoorLocalization/best_lr_ridge.csv"
        fileTestOutputLRLasso  = "../Predictions/IndoorLocalization/best_lr_lasso.csv"
        kfold = 5
        self.executeTrainLR(dataIndoor, kfold, alphaLst, fileTestOutputLRRidge, fileTestOutputLRLasso)
        
    
    #for kaggle competition power plant
    def predictDifferentModelsForPowerPlantKaggleComp(self):
        dataPowerPlant = self.readDataPowerPlant()
        trainX = dataPowerPlant[0]
        trainY = dataPowerPlant[1]
        testX = dataPowerPlant[2]
        
        #plotExploreDataPreTrain(trainX, trainY)            #plot train data
    
        #trainX = preprocessNANMethod(trainX)           #not working
        #trainX = preprocessTransform(trainX)           #not working
        #trainX = preprocessNormalize(trainX)           #not working
        #trainX = preprocessStandardScaler(trainX)      #might work
        #dataPowerPlant = (trainX, trainY, testX)
        
        print ("begin kaggle competition power plant DT;  Question 5 ----")
        depthLst = range(1, 40)               #range(1, 40) try different alpha from test
        lstRes = []
        for kfold in range(3, 15):
            fileTestOutputDT  = "../Predictions/PowerOutput/best_DT-competition" + str(kfold) + ".csv"
            (smallestMAE, kfold, bestDepth) = self.executeTrainDT(dataPowerPlant, kfold, depthLst, fileTestOutputDT)
            lstRes.append((smallestMAE, kfold, bestDepth))
        print ("power plant results of different MAE and kfold: ", sorted(lstRes, key = lambda x: (x[0], x[1], x[2])))
        
        '''
        #optimized cv kfold = 5-10, depth = 8 or 9
        kfold = 5
        depthLst = [8]
        fileTestOutputDT  = ""
        (smallestMAE, kfold, bestDepth) = self.executeTrainDT(dataPowerPlant, kfold, depthLst, fileTestOutputDT)
        '''

 #for kaggle competition indoor localization
    def predictDifferentModelsForIndoorLocalizationKaggleComp(self):
        dataIndoor = self.read_data_localization_indoors()
        
        trainX = dataIndoor[0]
        trainY = dataIndoor[1]
        testX = dataIndoor[2]
        #trainX = preprocessStandardScaler(trainX)      #might work
        #dataIndoor = (trainX, trainY, testX)
        print ("begin kaggle competition indoor localization KNN;  Question 5 ----")
        knnNeighbors = range(1, 40)  
        lstRes = []
        for kfold in range(2, 15):
            fileTestOutputKNN  = "../Predictions/IndoorLocalization/best_knn-competition" + str(kfold) + ".csv"
            (smallestMAE, kfold, bestNNeighbor) = self.executeTrainKNN(dataIndoor, kfold, knnNeighbors, fileTestOutputKNN)
            lstRes.append((smallestMAE, kfold, bestNNeighbor))
        
        print ("indoor localization KNN results of different MAE and kfold: ", sorted(lstRes, key = lambda x: (x[0], x[1], x[2])))
       
        '''
        depthLst = range(1, 30)               #range(1, 20) try different alpha from test
        lstRes = []
        for kfold in range(9, 20):
            fileTestOutputDT  = "../Predictions/IndoorLocalization/best_DT-competition" + str(kfold) + ".csv"
            (smallestMAE, kfold, bestDepth) = self.executeTrainDT(dataIndoor, kfold, depthLst, fileTestOutputDT)
            lstRes.append((smallestMAE, kfold, bestDepth))
        print ("indoor localization DT results of different MAE and kfold: ", sorted(lstRes, key = lambda x: (x[0], x[1], x[2])))
        '''


############################################################################

def main():
    
    regrHwObj = clsregressionHw()
    
    #for assigment querstion former part
    print (" -----begin regression for question 1-4 ---------")
    regrHwObj.predictDifferentModels()
        
    #for kaggle competition power plant
    print (" -----begin for question 5 kaggle competition for power plant ---------")
    regrHwObj.predictDifferentModelsForPowerPlantKaggleComp()
    
    #for kaggle competition indoor localization
    print (" -----begin for question 5 kaggle competition for Indoor localization ---------")
    regrHwObj.predictDifferentModelsForIndoorLocalizationKaggleComp()
    
    
if __name__== "__main__":
  main()
  


