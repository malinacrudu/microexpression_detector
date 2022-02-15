import cv2
import glob


def detect_eyes(img):
    # Read in the cascade classifiers for eyes
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eye_img = img.copy()
    eye_rect = eye_cascade.detectMultiScale(eye_img, scaleFactor=1.2, minNeighbors=6)
    if len(eye_rect) < 2:
        return
    min_1 = eye_rect[0][1]
    min_2 = eye_rect[1][1]
    eye_right = eye_rect[0]
    eye_left = eye_rect[1]
    for obj in eye_rect:
        if obj[1] < min_1:
            min_2 = min_1
            min_1 = obj[1]
            if obj[0] < eye_right[0]:
                eye_right = obj
            else:
                eye_left = obj
        if obj[1] < min_2:
            min_2 = obj[1]
            if obj[0] < eye_right[0]:
                eye_right = obj
            else:
                eye_left = obj

    max_width = max(eye_right[2], eye_left[2])
    start_y = min(eye_right[1], eye_left[1])
    end_y = start_y + max_width
    start_x = eye_right[0]
    end_x = eye_left[0] + max_width
    cropped = img[36:36 + 65, 11:119 + 65]
    # print(cropped)
    # for (x, y, w, h) in eye_rect:
    #     cv2.rectangle(eye_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
    return cropped


def haar_cascade():
    images = glob.glob('HaarCascade/Cropped-updated/Cropped/*/*/*.jpg', recursive=True)
    for imgPath in images:
        image = cv2.imread(imgPath)
        cropped = detect_eyes(image)
        if cropped is not None:
            cv2.imwrite(imgPath, cropped)
