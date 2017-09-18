# Import python modules
import numpy as np
import kaggle

from filesCommon import readTrainTestData
from preprocessing import preprocessNANMethod
from preprocessing import preprocessNormalize
from preprocessing import preprocessStandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

class clsregressionHw(object):
 
    
    def __init__(self):
      pass


    # Read in train and test data
    def readDataPowerPlant(self):
        
        #transferNPtoDataframe():
        print('Reading power plant dataset ...')
        fileNameTrain = '../../Data/PowerOutput/data_train.txt'
        fileNameTrainLabel = '../../Data/PowerOutput/labels_train.txt'
        fileTest = '../../Data/PowerOutput/data_test.txt'
        trainX, trainY, testX = readTrainTestData(fileNameTrain, fileNameTrainLabel, fileTest)
                
        print (" power data shape: ", trainX.shape, trainY.shape, testX.shape)

        return (trainX, trainY, testX)
    
    
    #execute power plant train to get model
    def executeTrainPowerPlant(self):
        trainX, trainY, testX = self.readDataPowerPlant()
        #trainX = preprocessNANMethod(trainX)
        #trainX = preprocessTransform(trainX)
        trainX = preprocessNormalize(trainX)
        #print ("train X: ", trainX)
        print ("train Y: ", trainY)
        
        #select the model 
        
        nNeighbors = 5
        k = 14
        kf = KFold(n_splits=k)
        i = 0
        averageMAE = 0.0
        sumMAE = 0.0
        for trainIndex, testIndex in kf.split(trainX):
            #print("TRAIN:", trainIndex, "TEST:", testIndex)
            xSplitTrain, XSplitTest = trainX[trainIndex], trainX[testIndex]
            ySplitTrain, ySplitTest = trainY[trainIndex], trainY[testIndex]
            
            neigh = KNeighborsRegressor(n_neighbors=nNeighbors)
            
            neigh.fit(xSplitTrain, ySplitTrain)
            
            #print ("parameter: ", neigh.get_params(deep=True))
            predYSplitTest = neigh.predict(XSplitTest)
            #print ("predYSplitTest : ", predYSplitTest)
            mAE =  self.computeMAEError(predYSplitTest, ySplitTest)
            print ("cv MAE error: ",i, mAE)
            
            sumMAE += mAE
            i +=1
        averageMAE  = sumMAE/k
        print ("averageMAE cv MAE error: ",averageMAE)
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
  


