import cv2
import glob
import math


def detect_eyes(img):
    # Read in the cascade classifiers for eyes
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eye_img = img.copy()
    eye_rect = eye_cascade.detectMultiScale(eye_img, 1.05, 3)

    if len(eye_rect) < 2:
        return

    min_1 = math.inf
    min_2 = math.inf
    eye_1 = []
    eye_2 = []
    # print(eye_rect)
    for obj in eye_rect:
        if obj[1] < min_1:
            min_2 = min_1
            min_1 = obj[1]
            eye_2 = eye_1
            eye_1 = obj
        if obj[1] < min_2 and obj[0] != eye_1[0]:
            min_2 = obj[1]
            eye_2 = obj
    # print("-----------------------")
    # print(eye_1, eye_2)
    max_width = max(eye_1[2], eye_2[2])
    start_y = min(eye_1[1], eye_2[1])
    end_y = start_y + max_width
    start_x = min(eye_1[0], eye_2[0])
    end_x = max(eye_1[0], eye_2[0]) + max_width
    # print(max_width)
    # print(start_y)
    # print(end_y)
    # print(start_x)
    # print(end_x)
    cropped = img[start_y:end_y, start_x:end_x]
    # print(cropped)
    # for (x, y, w, h) in eye_rect:
    #     cv2.rectangle(eye_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
    return cropped


def haar_cascade():
    images = glob.glob('Haar3/Cropped-updated/Cropped/*/*/*.jpg', recursive=True)
    nr = 0
    for imgPath in images:
        image = cv2.imread(imgPath)
        cropped = detect_eyes(image)
        if cropped is not None:
            cv2.imwrite(imgPath, cropped)
        else:
            cv2.imwrite('greseli3/' + str(nr) + '.jpg', image)
            nr += 1
    # image = cv2.imread('test.jpg')
    # cropped = detect_eyes(image)
    # cv2.imwrite('rez.jpg',cropped)
