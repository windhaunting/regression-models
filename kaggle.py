# kaggle.py module

import csv
import numpy as np

#Save prediction vector in Kaggle CSV format
#Input must be a Nx1, 1XN, or N long numpy vector
def kaggleize(predictions,file):

	if(len(predictions.shape)==1):
		predictions.shape = [predictions.shape[0],1]

	ids = 1 + np.arange(predictions.shape[0])[None].T
	kaggle_predictions = np.hstack((ids,predictions)).astype(int)
	writer = csv.writer(open(file, 'w'))
	writer.writerow(['# id','Prediction'])
	writer.writerows(kaggle_predictions)

