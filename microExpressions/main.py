from classifiers import *
from utils import *

import random

# get the data
input_set, output_set = createData()
# calculate histograms
X, y = calculateHistogramsWithLBPTOP(input_set, output_set)
# shuffle the data
temp = list(zip(X, y))
random.shuffle(temp)
X, y = zip(*temp)
X = np.array(X)
y = np.array(y)
log_message("--------------CASME WITH SOME SORT OF LBP-TOP--------------")
# train and test the model with the data
svm_model(X, y)
bayes_model(X, y)
neural_network_model(X, y)
decision_trees_model(X, y)

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# https://learnopencv.com/histogram-of-oriented-gradients/
