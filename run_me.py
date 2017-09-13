# Import python modules
import numpy as np
import kaggle

# Read in train and test data
def read_data_power_plant():
	print('Reading power plant dataset ...')
	train_x = np.loadtxt('../../Data/PowerOutput/data_train.txt')
	train_y = np.loadtxt('../../Data/PowerOutput/labels_train.txt')
	test_x = np.loadtxt('../../Data/PowerOutput/data_test.txt')

	return (train_x, train_y, test_x)

def read_data_localization_indoors():
	print('Reading indoor localization dataset ...')
	train_x = np.loadtxt('../../Data/IndoorLocalization/data_train.txt')
	train_y = np.loadtxt('../../Data/IndoorLocalization/labels_train.txt')
	test_x = np.loadtxt('../../Data/IndoorLocalization/data_test.txt')

	return (train_x, train_y, test_x)

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

############################################################################

train_x, train_y, test_x = read_data_power_plant()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

train_x, train_y, test_x = read_data_localization_indoors()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Create dummy test output values
predicted_y = np.ones(test_x.shape[0]) * -1
# Output file location
file_name = '../Predictions/IndoorLocalization/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

