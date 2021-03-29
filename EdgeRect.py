from transform import order_points
from transform import four_point_trainsform
import numpy as np
import cv2
import imutils
from imgEmhance import Enhancer


def preProcess(image):
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # the gray image
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    gray = cv2.Canny(gray, 75, 100)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.dilate(gray, kernel)
    screenCnt = 0
    cv2.imshow('edged', erosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
    cv2.findContours function return two value:
    edge about the image, it is a list
    the edge's attribute
    '''
    cnts = cv2.findContours(erosion.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(len(approx))
        if len(approx) == 4:
            screenCnt = approx
            break
    return screenCnt, ratio


if __name__ == '__main__':
    image = cv2.imread('test.jpg')
    height = image.shape[0]
    width = image.shape[1]
    # cv2.rectangle(image, (0,0), (width-10, height-10), (0,0,0),10)
    screenCnt, ratio = preProcess(image)

    warped = four_point_trainsform(image, screenCnt.reshape(4, 2) * ratio)
    enhancer = Enhancer()
    enhancerImg = enhancer.gamma(warped, 1.63)
    cv2.imshow('123', imutils.resize(image, height=500))
    cv2.imshow('edged', imutils.resize(enhancerImg, height=500))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
