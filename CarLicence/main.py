import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from skimage.transform import radon

# 空函数
def nothing(x):     
    pass

# 按原图比例修改图片大小
def MyResize(img, new_w):
    h, w = img.shape[:2]
    scale__ratio = float(new_w) / float(w) # 缩放比例
    new_h = int(h * scale__ratio)   # 新高度
    new_img = cv2.resize(image_origin, (new_w, new_h))  
    return new_img


if __name__ == '__main__':
    '''
    1.选择车牌图片
    '''
    # 显示窗口用于选择图像文件
    root = tk.Tk()
    root.withdraw()

    # 从打开的文件对话框中获取图像文件路径
    file_path = filedialog.askopenfilename(initialdir='汽车图片', title='选择图片', filetypes=(('JPEG', '*.jpg'), ('All files', '*.*')))

    # 读入图像
    image_origin = cv2.imread(file_path)

    # 修改图像大小
    image_RGB=  MyResize(image_origin, 500)
    image_GRAY = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)


    '''
    2.分割出车牌区域
    '''
    image_HSV = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2HSV)

    min_h = 0
    max_h = 23
    min_s = 109
    max_s = 255 
    min_v = 0
    max_v = 255

    # 阈值范围转化为numpy数组
    lower_range = np.array([min_h, min_s, min_v])
    upper_range = np.array([max_h, max_s, max_v])

    mask = cv2.inRange(image_HSV, lower_range, upper_range) # 掩膜
    # 位运算
    image_mask = cv2.bitwise_and(image_RGB, image_RGB, mask=mask)

    
    image_mask_gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY) # 灰度图像
    ret, image_mask_gray = cv2.threshold(image_mask_gray, 50, 100, cv2.THRESH_BINARY) # 二值化图像

    # 形态学
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(image_mask_gray, kernel, iterations=5) # 膨胀
    Close = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel) # 闭运算
    
    # 查找轮廓
    contours, hierarchy = cv2.findContours(Close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 查找最大轮廓
    max_area = 0
    max_contour = None
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > max_area:
            max_area = contour_area
            max_contour = contour

    # cv2.drawContours(image_RGB, [max_contour], -1, (0, 0, 255), 2)  # 绘制最大轮廓
    # cv2.imshow('image_RGB', image_RGB) 

    # 计算最小外接矩形
    rect = cv2.minAreaRect(max_contour)

    # 获取矩形的四个角点，并计算变换矩阵
    points = cv2.boxPoints(rect)
    dst_points = points.copy()
    dst_points[0][0] = points[3][0]
    dst_points[1][0] = points[2][0]
    dst_points[0][1] = points[1][1]
    dst_points[2][1] = points[3][1]
    M = cv2.getPerspectiveTransform(points, dst_points)
    # 进行透视变换
    h, w = image_RGB.shape[:2]
    iamge_rectify = cv2.warpPerspective(image_RGB, M, (w, h))  # 倾斜矫正的图像

    # 截取车牌图像
    x, y, w, h = cv2.boundingRect(max_contour) 
    license_rgb = iamge_rectify[y:y+h, x:x+w]  # 车牌图像
    # cv2.imshow('license_rgb', license_rgb)

    '''
    2.字符分割
    '''
    license_gray = cv2.cvtColor(license_rgb, cv2.COLOR_BGR2GRAY) # 车牌灰度图像

    # 1.去除边框
    _, license_threshold = cv2.threshold(license_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # 车牌图像二值化
    # cv2.imshow('license_threshold', license_threshold)

    h, w = license_gray.shape[:2]
    thresh = 0.3

    proj = np.sum(license_gray, axis=0)
    col_sum = np.where(proj > proj.max() * thresh)[0]
    proj = np.sum(license_gray, axis=1)
    row_sum = np.where(proj > proj.max() * thresh)[0]

    # 获取投影信息，以进行裁剪操作
    top, bottom = row_sum[0], row_sum[-1]
    left, right = col_sum[0], col_sum[-1]
 
    license_cut = license_threshold[top:bottom+1,left: right+1]     # 裁剪操作,得到裁剪后的图像
    license_cut = cv2.bitwise_not(license_cut) # 图像取反





 




    cv2.imshow('license_cut', license_cut) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

