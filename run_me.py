# Import python modules
import numpy as np
from kaggle import kaggleize

from filesCommon import readTrainTestData
from filesCommon import writeFileColumnwiseToKaggle

from preprocessing import preprocessNANMethod
from preprocessing import preprocessNormalize
from preprocessing import preprocessStandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

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
    def modelSelectionCV(self, trainX, trainY, modelFunc, *parameters):

        k = 10
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
        
    #execute power plant train to get model
    def executeTrainPowerPlant(self, fileTestOutput):
        trainX, trainY, testX = self.readDataPowerPlant()
        #trainX = preprocessNANMethod(trainX)
        #trainX = preprocessTransform(trainX)
        trainX = preprocessNormalize(trainX)
        #print ("train X: ", trainX)
        print ("train Y: ", trainY)

        knnNeighbors = range(1, 20)              #[1,2,3,4,5,6,7]
        i = 0
        smallestMAE = 1.0
        bestNNeighbor = 0
        for nNeighbor in knnNeighbors:
            averageMAE = self.modelSelectionCV(trainX, trainY, KNeighborsRegressor, nNeighbor)
            i += 1
            print ("averageMAE cv MAE error: ", averageMAE)
            if averageMAE < smallestMAE:
                smallestMAE = averageMAE
                bestNNeighbor = nNeighbor
        
        print (" bestNNeighbor: ", bestNNeighbor)
        predY = trainTestWholeData(trainX, trainY, testX, KNeighborsRegressor, bestNNeighbor)
        
        #output to file
        fileTestOutput , columnNameLst, columnsValues):
        writeFileColumnwiseToKaggle()
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


    
    regrHwObj.executeTrainPowerPlant()
    
    
    
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
  


