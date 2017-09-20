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
    

     
    
    # select model parameter by using CV
    def modelSelectionCV(self, trainX, trainY, k, modelFunc, *parameters):

        kf = KFold(n_splits=k)
        averageMAE = 0.0
        sumMAE = 0.0
        for trainIndex, testIndex in kf.split(trainX):
            #print("TRAIN:", trainIndex, "TEST:", testIndex)
            xSplitTrain, XSplitTest = trainX[trainIndex], trainX[testIndex]
            ySplitTrain, ySplitTest = trainY[trainIndex], trainY[testIndex]
            
            #neigh = KNeighborsRegressor(n_neighbors=nNeighbor)
            model =  modelFunc(*parameters)
            model.fit(xSplitTrain, ySplitTrain)
            
            #print ("parameter: ", neigh.get_params(deep=True))
            predYSplitTest = model.predict(XSplitTest)
            #print ("predYSplitTest : ", predYSplitTest)
            mAE =  self.computeMAEError(predYSplitTest, ySplitTest)
            #print ("cv MAE error: ",i, mAE)
            sumMAE += mAE

        averageMAE  = sumMAE/k
        return averageMAE
    
    # use whole train data to do train and then test
    def trainTestWholeData(self, trainX, trainY, testX, modelFunc, *parameters):
        model =  modelFunc(*parameters)
        model.fit(trainX, trainY)
            
        #print ("parameter: ", neigh.get_params(deep=True))
        predY = model.predict(testX)
        
        return predY
        
    #execute power plant train to get model parameter of KNN
    def executeTrainPowerPlantKNN(self, fileTestOutputKNN):
        trainX, trainY, testX = self.readDataPowerPlant()
        #trainX = preprocessNANMethod(trainX)           #not working
        #trainX = preprocessTransform(trainX)           #not working
        #trainX = preprocessNormalize(trainX)           #not working
        #trainX = preprocessStandardScaler(trainX)      #might work
        #print ("train X: ", trainX)
        print ("train Y: ", trainY)


        #use  k nearest neighbor knn
        knnNeighbors = range(1, 30)    #len(trainX), 2)              #[1,2,3,4,5,6,7]
        i = 0
        smallestMAE = 1.0
        bestNNeighbor = 0
        for nNeighbor in knnNeighbors:
            k = 10
            averageMAE = self.modelSelectionCV(trainX, trainY, k, KNeighborsRegressor, nNeighbor)
            i += 1
            print ("averageMAE cv MAE error: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestMAE = averageMAE
                bestNNeighbor = nNeighbor
        
        print (" bestNNeighbor: ", bestNNeighbor)
        predY = self.trainTestWholeData(trainX, trainY, testX, KNeighborsRegressor, bestNNeighbor)
        print ("predY : ", predY)
        #output to file
        kaggleize(predY, fileTestOutputKNN)
        
       
        
    #execute linear regression powerPlant      
    def executeTrainPowerPlantLR(self, fileTestOutputLRRidge, fileTestOutputLRLasso):
        trainX, trainY, testX = self.readDataPowerPlant()
        
        alphaLst = [1e-6, 1e-4, 1e-2, 1, 10]              #try different alpha from test

        smallestMAE = 1.0
        bestAlpha = 0
        
        for alpha in alphaLst:
            k = 10
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
        
        
        smallestMAE = 1.0
        bestAlpha = 0
        
        for alpha in alphaLst:
            k = 10
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
    def executeTrainPowerPlantDT(self, fileTestOutputDT):
        trainX, trainY, testX = self.readDataPowerPlant()
        
        depthLst = [3, 6, 9, 12, 15]              #range(1, 20) try different alpha from test

        smallestMAE = 1.0
        bestDepth = 0
        
        for depth in depthLst:
            k = 10
            averageMAE = self.modelSelectionCV(trainX, trainY, k, DecisionTreeRegressor, depth)
            print ("averageMAE cv MAE error: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestMAE = averageMAE
                bestDepth = depth
        
        print (" bestDepth: ", bestDepth)
        predY = self.trainTestWholeData(trainX, trainY, testX, DecisionTreeRegressor, bestDepth)
        print ("predY : ", predY)
        #output to file
        kaggleize(predY, fileTestOutputDT)
    
    # read in train and test data of indoor locationzation
    def read_data_localization_indoors(self):
        x = 1
    
   
    # Compute MAE
    def computeMAEError(self, y_hat, y):
        	# mean absolute error
        return np.abs(y_hat - y).mean()





############################################################################


def main():
    
    regrHwObj = clsregressionHw()


    fileTestOutputKNN  = "../Predictions/PowerOutput/best_knn.csv"
    regrHwObj.executeTrainPowerPlantKNN(fileTestOutputKNN)
    
    fileTestOutputLRRidge  = "../Predictions/PowerOutput/best_lr_ridge.csv"
    fileTestOutputLRLasso  = "../Predictions/PowerOutput/best_lr_lasso.csv"
    regrHwObj.executeTrainPowerPlantLR(fileTestOutputLRRidge, fileTestOutputLRLasso)
    
    fileTestOutputDT  = "../Predictions/PowerOutput/best_DT.csv"
    regrHwObj.executeTrainPowerPlantDT(fileTestOutputDT)


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
  


