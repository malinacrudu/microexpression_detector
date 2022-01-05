from utils import *
import random

input_set, output_set = createData()
y = output_set
X = calculateHistograms(input_set)
temp = list(zip(X, y))
random.shuffle(temp)
X, y = zip(*temp)
X = np.array(X)
y = np.array(y)
svm_model(X, y)
