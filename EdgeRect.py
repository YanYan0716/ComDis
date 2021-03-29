# from transform import order_points
# from transform import four_point_trainsform
# import numpy as np
# import cv2
# import imutils
# from imgEmhance import Enhancer
#
#
# def preProcess(image):
#     ratio = image.shape[0] / 500.0
#     image = imutils.resize(image, height=500)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # the gray image
#     gray = cv2.GaussianBlur(gray, (1, 1), 0)
#     gray = cv2.Canny(gray, 75, 100)
#     kernel = np.ones((3, 3), np.uint8)
#     erosion = cv2.dilate(gray, kernel)
#     screenCnt = 0
#     cv2.imshow('edged', erosion)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     '''
#     cv2.findContours function return two value:
#     edge about the image, it is a list
#     the edge's attribute
#     '''
#     cnts = cv2.findContours(erosion.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0]
#     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
#     for c in cnts:
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#         print(len(approx))
#         if len(approx) == 4:
#             screenCnt = approx
#             break
#     return screenCnt, ratio
#
#
# if __name__ == '__main__':
#     image = cv2.imread('test.jpg')
#     height = image.shape[0]
#     width = image.shape[1]
#     # cv2.rectangle(image, (0,0), (width-10, height-10), (0,0,0),10)
#     screenCnt, ratio = preProcess(image)
#
#     warped = four_point_trainsform(image, screenCnt.reshape(4, 2) * ratio)
#     enhancer = Enhancer()
#     enhancerImg = enhancer.gamma(warped, 1.63)
#     cv2.imshow('123', imutils.resize(image, height=500))
#     cv2.imshow('edged', imutils.resize(enhancerImg, height=500))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
import cv2 as cv
import numpy as np

# 读入图片
src = cv.imread('test.jpg')
# 转换成灰度图
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# 二值化
ret, thresh = cv.threshold(gray, 129, 255, cv.THRESH_BINARY)

# 查找轮廓
# binary-二值化结果，contours-轮廓信息，hierarchy-层级
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# 显示轮廓，tmp为黑色的背景图
tmp = np.zeros(src.shape, np.uint8)
res = cv.drawContours(tmp, contours, -1, (250, 255, 255), 1)
cv.imshow('Allcontours', res)
for c in contours[1:]:
    x, y, w, h = cv.boundingRect(c)  # 外接矩形
    cv.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)
#
# cnt = contours[8]
# tmp2 = np.zeros(src.shape, np.uint8)
# res2 = cv.drawContours(tmp2, cnt, -1, (250, 255, 255), 2)
# cv.imshow('cnt', res2)
#
# # 轮廓特征
# # 面积
# print(cv.contourArea(cnt))
# # 周长,第二个参数指定图形是否闭环,如果是则为True, 否则只是一条曲线.
# print(cv.arcLength(cnt, True))
#
# # 轮廓近似，epsilon数值越小，越近似
# epsilon = 0.08 * cv.arcLength(cnt, True)
# approx = cv.approxPolyDP(cnt, epsilon, True)
# tmp2 = np.zeros(src.shape, np.uint8)
# # 注意，这里approx要加中括号
# res3 = cv.drawContours(tmp2, [approx], -1, (250, 250, 255), 1)
# cv.imshow('approx', res3)
#
# # 外接图形
# x, y, w, h = cv.boundingRect(cnt)
# # 直接在图片上进行绘制，所以一般要将原图复制一份，再进行绘制
# tmp3 = src.copy()
# res4 = cv.rectangle(tmp3, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv.imshow('rectangle', res)

cv.waitKey()
