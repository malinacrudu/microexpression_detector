import glob
import sys
from datetime import date
import cv2
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from LBP import LocalBinaryPatterns

np.set_printoptions(threshold=sys.maxsize)


# a function that gets the input and output
# for both problems: facial recognition and emotion detection
def getDataChildren(path):
    images = glob.glob(path, recursive=True)
    inputEmo = []
    outputEmo = []
    for image in images:
        inputEmo.append(image)
        if 'surprise' in image:
            outputEmo.append('surprise')
        if 'sad' in image:
            outputEmo.append('sadness')
        if 'neutral' in image:
            outputEmo.append('others')
        if 'happy' in image:
            outputEmo.append('happiness')
        if 'fear' in image:
            outputEmo.append('fear')
        if 'disgust' in image:
            outputEmo.append('disgust')
        if 'angry' in image:
            outputEmo.append('others')
    return inputEmo, outputEmo


# a function that gets the input and output
# for both problems: facial recognition and emotion detection
def getDataCK(path):
    images = glob.glob(path, recursive=True)
    inputEmo = []
    outputEmo = []
    for image in images:
        inputEmo.append(image)
        if 'anger' in image:
            outputEmo.append('others')
        if 'contempt' in image:
            outputEmo.append('others')
        if 'disgust' in image:
            outputEmo.append('disgust')
        if 'fear' in image:
            outputEmo.append('fear')
        if 'happy' in image:
            outputEmo.append('happiness')
        if 'surprise' in image:
            outputEmo.append('surprise')
        if 'sadness' in image:
            outputEmo.append('sadness')
    return inputEmo, outputEmo


# a function that creates a map where for each folder
# (in each folder it is a micro-expression represented as a
# set of images which are frames extracted from a video)
# associates the emotion displayed
def getClassification():
    labels = {}
    with open('CASME2-coding-20190701.csv', mode='r')as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split(',')
            labels[line[1]] = line[-1]
    return labels


# a function that gets all the images and creates the input and output sets
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


# a function that takes all the input images
# processes them and saves them
def processData():
    images = glob.glob('Cropped-updated/Cropped/*/*/*.jpg', recursive=True)
    for imgPath in images:
        image = cv2.imread(imgPath)
        new_image = processImage(image, (200, 250))
        cv2.imwrite(imgPath, new_image)


# a function that divides the data into training and validation
# with a ration of 80:20
def divideData(input, output):
    np.random.seed(5)
    indexes = [i for i in range(len(input))]
    trainSample = np.random.choice(indexes, int(0.8 * len(input)), replace=False)
    testSample = [i for i in indexes if i not in trainSample]

    trainingInputSet = [input[i] for i in trainSample]
    trainingOutputSet = [output[i] for i in trainSample]
    validationInputSet = [input[i] for i in testSample]
    validationOutputSet = [output[i] for i in testSample]

    return trainingInputSet, trainingOutputSet, validationInputSet, validationOutputSet


# a function that given an image
# applies a black and white filter over it
# and resizes it a specific size given as a tuple
def processImage(img, dimTuple):
    face_img = img.copy()
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, dimTuple)
    return gray


# a function that creates an LBP histogram for each black and white
# image corresponding to the element in the input set
def calculateHistograms(inputSet, resizeDim=(320, 320), necessaryResize=False):
    desc = LocalBinaryPatterns(8, 1, 48)
    data = []
    for i in range(len(inputSet)):
        print(i)
        imagePath = inputSet[i]
        image = cv2.imread(imagePath)
        if necessaryResize:
            image = cv2.resize(image, resizeDim)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(image)
        data.append(hist)
    return data


# having the input and output data it removes the
# data that has "others" as a category for emotions
def createData():
    input_set, output_set = getData()
    new_input = []
    new_output = []
    for i in range(len(input_set)):
        if i>5:
            break
        if output_set[i] != "others":
            new_input.append(input_set[i])
            new_output.append(output_set[i])
    new_output = oneHotEncoding(new_output)
    return new_input, new_output


def createDataCK():
    input_set, output_set = getDataCK(
        "C:\\Users\\Utilizator\\Desktop\\anul3\\Sem1\\Calcul afectiv\\microexpression_detector\\microExpressions\\CK+48\\*\\*.png")
    new_input = []
    new_output = []
    for i in range(len(input_set)):
        if output_set[i] != "others":
            new_input.append(input_set[i])
            new_output.append(output_set[i])
    new_output = oneHotEncoding(new_output)
    print(len(new_input))
    print(len(new_output))
    return new_input, new_output


def createDataChildren():
    input_set, output_set = getDataChildren(
        "C:\\Users\\Utilizator\\Desktop\\anul3\\Sem1\\Calcul afectiv\\microexpression_detector\\microExpressions\\poze_procesate\\*.jpg")
    new_input = []
    new_output = []
    for i in range(len(input_set)):
        if output_set[i] != "others":
            new_input.append(input_set[i])
            new_output.append(output_set[i])
    new_output = oneHotEncoding(new_output)
    print(len(new_input))
    print(len(new_output))
    return new_input, new_output


# creates the vector based on One Hot Encoding given the
# outputLabels vector
def oneHotEncoding(outputLabels):
    label = LabelEncoder()
    int_data = label.fit_transform(outputLabels)
    print(label.classes_)
    int_data = int_data.reshape(len(int_data), 1)
    onehot_data = OneHotEncoder(sparse=False)
    onehot_data = onehot_data.fit_transform(int_data)
    onehot_data = np.argmax(onehot_data, axis=1)
    return onehot_data


# logs a message in the logs file with a specific message and current date
def log_message(data):
    f = open("logs_decision_trees.txt", "a")
    f.write(str(data) + " " + str(date.today()) + "\n")
    f.close()


def writeToCsv():
    input_set, output_set = createData()
    data = calculateHistograms(input_set, output_set)
    print(len(data))
    # import csv
    # filename = "data.csv"
    # print(len(data))
    # with open(filename, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(data)


# a function that creates and saves a histogram
def makeHistogram(labels):
    N, bins, patches = plt.hist(labels, bins=5, edgecolor='white', linewidth=1)
    patches[0].set_facecolor('tomato')
    patches[1].set_facecolor('greenyellow')
    patches[1].set_facecolor('mediumorchid')
    patches[2].set_facecolor('gold')
    patches[3].set_facecolor('sandybrown')
    patches[4].set_facecolor('cornflowerblue')
    plt.title('Class distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Number of samples')
    plt.savefig("dataset.png")
    plt.show()


# a function that maps every digit from 0 to 4 to a different emotion from
# the set of labels used in our classification
def fromDigitToEmotion(digit):
    if digit == 0:
        return 'disgust'
    elif digit == 1:
        return 'happiness'
    if digit == 2:
        return 'repression'
    if digit == 3:
        return 'sadness'
    if digit == 4:
        return 'surprise'


# a function that creates a confusion matrix
def createConfusionMatrix(labelsComputed, labelsTrue):
    labelC = []
    labelV = []
    for i in range(len(labelsComputed)):
        labelC.append(fromDigitToEmotion(labelsComputed[i]))
        labelV.append(fromDigitToEmotion(labelsTrue[i]))
    data = {
        'y_Actual': labelV,
        'y_Predicted': labelC
    }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True, cmap="Pastel2")
    plt.savefig("confusion.png")


def getDataKarolinska(path):
    # https://www.kdef.se/home/aboutKDEF.html
    images = glob.glob(path, recursive=True)
    inputEmo = []
    outputEmo = []
    for image in images:
        inputEmo.append(image)
        if 'SU' in image:
            outputEmo.append('surprise')
        if 'SA' in image:
            outputEmo.append('sadness')
        if 'NE' in image:
            outputEmo.append('others')
        if 'HA' in image:
            outputEmo.append('happiness')
        if 'AF' in image:
            outputEmo.append('fear')
        if 'AN' in image:
            outputEmo.append('others')
        if 'DI' in image:
            outputEmo.append('disgust')
    return inputEmo, outputEmo


def createDataKarolinska():
    input_set, output_set = getDataKarolinska("C:\\Users\\Utilizator\\Desktop\\anul3\\Sem1\\Calcul afectiv\\microexpression_detector\\microExpressions\\KDEF (reeks A) zonder haarlijn\\*.jpg")
    new_input = []
    new_output = []
    for i in range(len(input_set)):
        if output_set[i] != "others":
            new_input.append(input_set[i])
            new_output.append(output_set[i])
    new_output = oneHotEncoding(new_output)
    print(len(new_input))
    print(len(new_output))
    print(new_output)
    return new_input, new_output
