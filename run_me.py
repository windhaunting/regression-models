# Import python modules
import numpy as np
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
from sklearn.model_selection import cross_val_score

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
                
        print (" power data shape: ", trainX.shape, trainY.shape, testX.shape)

        return (trainX, trainY, testX)
    

    # read in train and test data of indoor locationzation
    def read_data_localization_indoors(self):
        print('Reading indoor localization dataset ...')
        fileNameTrain = '../../Data/IndoorLocalization/data_train.txt'
        fileNameTrainLabel = '../../Data/IndoorLocalization/labels_train.txt'
        fileTest = '../../Data/IndoorLocalization/data_test.txt'
        trainX, trainY, testX = readTrainTestData(fileNameTrain, fileNameTrainLabel, fileTest)
                
        print (" IndoorLocalization data shape: ", trainX.shape, trainY.shape, testX.shape)

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
            #print ("predYSplitTest : ", predYSplitTest)
            mAE =  self.computeMAEError(predYSplitTest, ySplitTest)
            #print ("cv MAE error: ",i, mAE)
            sumMAE += mAE

        averageMAE  = sumMAE/kfold
        return averageMAE
    
    #use cros_val_score
    def modelSelectionCVCrosValScore(self, trainX, trainY, kfold, modelFunc, *args):

        model =  modelFunc(*args)
            
        scoresLst = cross_val_score(model, trainX, trainY, scoring="neg_mean_absolute_error", cv=5, n_jobs=4)
        averageMAE = np.mean(scoresLst)
       
        return abs(averageMAE)
    # use whole train data to do train and then test
    def trainTestWholeData(self, trainX, trainY, testX, modelFunc, *args):
        model =  modelFunc(*args)
        model.fit(trainX, trainY)
            
        #print ("parameter: ", neigh.get_params(deep=True))
        predY = model.predict(testX)
        
        return predY
    
    
    #execute power plant train to get model parameter of KNN
    def executeTrainPowerPlantKNN(self, data, knnNeighbors, fileTestOutputKNN):
        trainX = data[0]
        trainY = data[1]
        testX = data[2]
 
        #print ("train X: ", trainX.shape)


        #use  k nearest neighbor knn
        i = 0
        smallestMAE = 2^32
        bestNNeighbor = knnNeighbors[0]
        for nNeighbor in knnNeighbors:
            k = 5      #cv kfold value
            averageMAE = self.modelSelectionCV(trainX, trainY, k, KNeighborsRegressor, nNeighbor)
            i += 1
            print ("averageMAE cv MAE error KNN: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestMAE = averageMAE
                bestNNeighbor = nNeighbor
        
        print (" bestNNeighbor KNN: ", bestNNeighbor)
        predY = self.trainTestWholeData(trainX, trainY, testX, KNeighborsRegressor, bestNNeighbor)
        print ("predY : KNN", predY)
        #output to file
        kaggleize(predY, fileTestOutputKNN)
        
    
    #execute linear regression powerPlant      
    def executeTrainPowerPlantLR(self, data, alphaLst, fileTestOutputLRRidge, fileTestOutputLRLasso):
        trainX = data[0]
        trainY = data[1]
        testX = data[2]

        #trainX = preprocessNormalize(trainX)           #not working
        #trainX = preprocessStandardScaler(trainX)      #might work
        
        #ridge begins
        smallestMAE = 2^32
        bestAlpha = alphaLst[0]
    
        for alpha in alphaLst:
            k = 5
            averageMAE = self.modelSelectionCV(trainX, trainY, k, Ridge, alpha)
            print ("averageMAE cv MAE error Ridge: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestMAE = averageMAE
                bestAlpha = alpha
        
        print (" bestAlpha Ridge: ", bestAlpha)
        predY = self.trainTestWholeData(trainX, trainY, testX, Ridge, bestAlpha)
        print ("predY Ridge: ", predY)
        #output to file
        kaggleize(predY, fileTestOutputLRRidge)
        
        #lasso begins
        smallestMAE = 2^32
        bestAlpha = alphaLst[0]
        
        for alpha in alphaLst:
            k = 5
            averageMAE = self.modelSelectionCV(trainX, trainY, k, Lasso, alpha)
            print ("averageMAE cv MAE error Lasso: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestMAE = averageMAE
                bestAlpha = alpha
        
        print (" bestAlpha Lasso: ", bestAlpha)
        predY = self.trainTestWholeData(trainX, trainY, testX, Lasso, bestAlpha)
        print ("predY Lasso: ", predY)
        #output to file
        kaggleize(predY, fileTestOutputLRLasso)
        
    
    
    #execute Decision tree powerPlant      
    def executeTrainPowerPlantDT(self, data, depthLst, fileTestOutputDT):
        trainX = data[0]
        trainY = data[1]
        testX = data[2]
        
        smallestMAE = 2^32
        bestDepth = depthLst[0]
        
        for depth in depthLst:
            k = 5
            args = ("mae", "best", depth)            # {"criterion": "mae", "splitter": "best", "max_depth": depth} 
            #averageMAE = self.modelSelectionCV(trainX, trainY, k, DecisionTreeRegressor, *args)
            averageMAE = self.modelSelectionCVCrosValScore(trainX, trainY, k, DecisionTreeRegressor, *args)

            print ("averageMAE cv MAE error DT: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestMAE = averageMAE
                bestDepth = depth
        args = ("mae", "best", bestDepth)            # {"criterion": "mae", "splitter": "best", "max_depth": bestDepth} 
        print (" bestDepth DT: ", bestDepth, smallestMAE)
        predY = self.trainTestWholeData(trainX, trainY, testX, DecisionTreeRegressor, *args)
        #print ("predY DT: ", predY)
        #output to file
        kaggleize(predY, fileTestOutputDT)
    

     #for assignment questions:
    def predictDifferentModels(self):
        #predict power plant here
        #dataPowerPlant = self.readDataPowerPlant()
       
        #knn begins
        knnNeighbors = [3,5,10,20,25]   #range(1, 30)    #len(trainX), 2)              
        fileTestOutputKNN  = "../Predictions/PowerOutput/best_knn.csv"
        #self.executeTrainPowerPlantKNN(dataPowerPlant, knnNeighbors, fileTestOutputKNN)
        
        
        #linear regression begins
        alphaLst = [1e-6, 1e-4, 1e-2, 1, 10]              #try different alpha from test
        fileTestOutputLRRidge  = "../Predictions/PowerOutput/best_lr_ridge.csv"    
        fileTestOutputLRLasso  = "../Predictions/PowerOutput/best_lr_lasso.csv"    
        #self.executeTrainPowerPlantLR(dataPowerPlant, alphaLst, fileTestOutputLRRidge, fileTestOutputLRLasso)
        
        # Decision tree begins
        depthLst = [3, 6, 9, 12, 15]              #range(1, 20) try different alpha from test
        fileTestOutputDT  = "../Predictions/PowerOutput/best_DT.csv"
        #self.executeTrainPowerPlantDT(dataPowerPlant, depthLst, fileTestOutputDT)
    
        
        # predict for indoor localization here
        dataIndoor = self.read_data_localization_indoors()
        
        #knn begins
        knnNeighbors = [3,5,10,20,25]   #range(1, 30)    #len(trainX), 2) 
        fileTestOutputKNN  = "../Predictions/IndoorLocalization/best_knn.csv"
        #self.executeTrainPowerPlantKNN(dataIndoor, knnNeighbors, fileTestOutputKNN)
        
        #linear regression begins
        alphaLst = [1e-6, 1e-4, 1e-2, 1, 10]              #try different alpha from test
        fileTestOutputLRRidge  = "../Predictions/IndoorLocalization/best_lr_ridge.csv"
        fileTestOutputLRLasso  = "../Predictions/IndoorLocalization/best_lr_lasso.csv"
        #self.executeTrainPowerPlantLR(dataIndoor, alphaLst, fileTestOutputLRRidge, fileTestOutputLRLasso)
        
        # Decision tree begins
        depthLst = [3, 6, 9, 12, 15]              #range(1, 20) try different alpha from test
        fileTestOutputDT  = "../Predictions/IndoorLocalization/best_DT.csv"
        self.executeTrainPowerPlantDT(dataIndoor, depthLst, fileTestOutputDT)
    
    
    #for kaggle competition
    def predictDifferentModelsForKaggleComp(self):
        dataPowerPlant = self.readDataPowerPlant()
        trainX = dataPowerPlant[0]
        trainY = dataPowerPlant[1]
        testX = dataPowerPlant[2]
        
        #trainX = preprocessNANMethod(trainX)           #not working
        #trainX = preprocessTransform(trainX)           #not working
        #trainX = preprocessNormalize(trainX)           #not working
        #trainX = preprocessStandardScaler(trainX)      #might work
        depthLst = range(1, 20)             #range(1, 20) try different alpha from test
        fileTestOutputDT  = "../Predictions/PowerOutput/best_DT-competition.csv"
        self.executeTrainPowerPlantDT(dataPowerPlant, depthLst, fileTestOutputDT)
    
    
############################################################################

def main():
    
    regrHwObj = clsregressionHw()
    
    #for assigment querstion former part
    #regrHwObj.predictDifferentModels()
    
    
    #for kaggle competition later pater
    regrHwObj.predictDifferentModelsForKaggleComp()
    
    
    '''
    train_x, train_y, test_x = regrHwObj.readDataPowerPlant()
    print('Train=', train_x.shape)
    print('Test=', test_x.shape)
    
    # Create dummy test output values
    predicted_y = np.ones(test_x.shape[0]) * -1
    # Output file location
    file_name = '../Predictions/IndoorLocalization/best.csv'
    # Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(predicted_y, file_name)
    '''
    


if __name__== "__main__":
  main()
  


