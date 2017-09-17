# Import python modules
import numpy as np
import kaggle

from filesCommon import readTrainTestData
from preprocessing import preprocessNANMethod
class clsregressionHw(object):
 
    
    def __init__(self):
      pass


    # Read in train and test data
    def read_data_power_plant(self):
        
        #transferNPtoDataframe():
        print('Reading power plant dataset ...')
        fileNameTrain = '../../Data/PowerOutput/data_train.txt'
        fileNameTrainLabel = '../../Data/PowerOutput/labels_train.txt'
        fileTest = '../../Data/PowerOutput/data_test.txt'
        train_x, train_y, test_x = readTrainTestData(fileNameTrain, fileNameTrainLabel, fileTest)
                
        print (" power data shape: ", train_x.shape, train_y.shape, test_x.shape)

        return (train_x, train_y, test_x)
    
    
    
    def executeTrain(self):
        x = 1
        train_x, train_y, test_x = self.read_data_power_plant()
        preprocessNANMethod(train_x)
    
    def read_data_localization_indoors(self):
        x = 1
    
    # Compute MAE
    def compute_error(y_hat, y):
        	# mean absolute error
        return np.abs(y_hat - y).mean()

############################################################################



def main():
    
    regrHwObj = clsregressionHw()

    regrHwObj.executeTrain()
    
    '''
    train_x, train_y, test_x = regrHwObj.read_data_localization_indoors()
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
  


