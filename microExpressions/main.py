from classifiers import *
from utils import *
import random

# get the data
input_set, output_set = createDataChildren()
y = output_set
# calculate histograms
X = calculateHistograms(input_set)
# shuffle the data
temp = list(zip(X, y))
random.shuffle(temp)
X, y = zip(*temp)
X = np.array(X)
y = np.array(y)
log_message("--------------TESTE PE CAFE CU OTHERS PT NEUTRAL AND ANGRY AND 0 REPRESSION--------------")
# train and test the model with the data
svm_model(X, y)
# bayes_model(X, y)
# neural_network_model(X, y)


# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
