from utils import *
import random

# get the data
input_set, output_set = createData()
y = output_set
# calculate histograms
X = calculateHistograms(input_set)
# shuffle the data
temp = list(zip(X, y))
random.shuffle(temp)
X, y = zip(*temp)
X = np.array(X)
y = np.array(y)
# train and test the model with the data
svm_model(X, y)
