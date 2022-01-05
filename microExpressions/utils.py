import glob
import sys
from datetime import date
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from LBP import LocalBinaryPatterns

np.set_printoptions(threshold=sys.maxsize)


def getClassification():
    labels = {}
    with open('CASME2-coding-20190701.csv', mode='r')as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split(',')
            labels[line[1]] = line[-1]
    return labels


def getData():
    images = glob.glob('Cropped-updated/Cropped/*/*/*.jpg', recursive=True)
    input_set = []
    output_set = []
    labels = getClassification()
    for img in images:
        imgPath = img.split('\\')
        input_set.append(img)
        output_set.append(labels[imgPath[2]])
    return input_set, output_set


def processData():
    images = glob.glob('Cropped-updated/Cropped/*/*/*.jpg', recursive=True)
    for imgPath in images:
        image = cv2.imread(imgPath)
        new_image = adjusted_detect_face(image, (200, 250))
        cv2.imwrite(imgPath, new_image)


def divideData(input, output):
    np.random.seed(5)
    indexes = [i for i in range(len(input))]
    trainSample = np.random.choice(indexes, int(0.8 * len(input)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainingInputSet = [input[i] for i in trainSample]
    trainingOutputSet = [output[i] for i in trainSample]
    validationInputSet = [input[i] for i in testSample]
    validationOutputSet = [output[i] for i in testSample]

    return trainingInputSet, trainingOutputSet, validationInputSet, validationOutputSet


def adjusted_detect_face(img, dimTuple):
    face_img = img.copy()
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, dimTuple)
    return gray


def calculateHistograms(trainingInputSet):
    desc = LocalBinaryPatterns(8, 1, 50)
    data = []
    for i in range(len(trainingInputSet)):
        print(i)
        imagePath = trainingInputSet[i]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(image)
        data.append(hist)
    return data


def createData():
    input_set, output_set = getData()
    new_input = []
    new_output = []
    for i in range(len(input_set)):
        if output_set[i] != "others":
            new_input.append(input_set[i])
            new_output.append(output_set[i])
    new_output = oneHotEncoding(new_output)
    return new_input, new_output


def svm_model(X, y):
    trials = 20
    while trials:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        from sklearn import metrics
        log_message("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + " Trial: " + str(trials))
        trials -= 1


def oneHotEncoding(categories):
    label = LabelEncoder()
    int_data = label.fit_transform(categories)
    int_data = int_data.reshape(len(int_data), 1)
    onehot_data = OneHotEncoder(sparse=False)
    onehot_data = onehot_data.fit_transform(int_data)
    onehot_data = np.argmax(onehot_data, axis=1)
    print(onehot_data)
    return onehot_data


def log_message(data):
    f = open("logs.txt", "a")
    f.write(str(data) + " " + str(date.today()) + "\n")
    f.close()
