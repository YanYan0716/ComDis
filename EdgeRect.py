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
import cv2
import numpy as np
img = cv2.pyrDown(cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED))
# threshold 函数对图像进行二化值处理，由于处理后图像对原图像有所变化，因此img.copy()生成新的图像，cv2.THRESH_BINARY是二化值
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
# findContours函数查找图像里的图形轮廓
# 函数参数thresh是图像对象
# 层次类型，参数cv2.RETR_EXTERNAL是获取最外层轮廓，cv2.RETR_TREE是获取轮廓的整体结构
# 轮廓逼近方法
# 输出的返回值，image是原图像、contours是图像的轮廓、hier是层次类型
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # 轮廓绘制方法一
    # boundingRect函数计算边框值，x，y是坐标值，w，h是矩形的宽和高
    x, y, w, h = cv2.boundingRect(c)
    # 在img图像画出矩形，(x, y), (x + w, y + h)是矩形坐标，(0, 255, 0)设置通道颜色，2是设置线条粗度
    cv2.rectangle (img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 轮廓绘制方法二
# 查找最小区域
rect = cv2.minAreaRect(c)
# 计算最小面积矩形的坐标
box = cv2.boxPoints(rect)
# 将坐标规范化为整数
box = np.int0(box)
# 绘制矩形
cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

# 轮廓绘制方法三
# 圆心坐标和半径的计算
(x, y), radius = cv2.minEnclosingCircle(c)
# 规范化为整数
center = (int(x), int(y))
radius = int(radius)
# 勾画圆形区域
img = cv2.circle (img, center, radius, (0, 255, 0), 2)

# # 轮廓绘制方法四
# 围绕图形勾画蓝色线条
cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
# 显示图像
cv2.imshow("contours", img)
cv2.waitKey()
cv2.destroyAllWindows()
