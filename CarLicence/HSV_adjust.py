import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 空函数
def nothing(x):     
    pass

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
    # 按原图比例修改图片大小
    h, w = image_origin.shape[:2]
    new_w = 500         # 新宽度
    scale__ratio = float(new_w) / float(w) # 缩放比例
    new_h = int(h * scale__ratio)   # 新高度
    image_RGB = cv2.resize(image_origin, (new_w, new_h))  
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

    cv2.namedWindow('HSV_adjust') # 创建1个窗口调整HSV阈值
    # 创建滑块
    cv2.createTrackbar('min_h', 'HSV_adjust', min_h, 255, nothing)
    cv2.createTrackbar('max_h', 'HSV_adjust', max_h, 255, nothing)
    cv2.createTrackbar('min_s', 'HSV_adjust', min_s, 255, nothing)
    cv2.createTrackbar('max_s', 'HSV_adjust', max_s, 255, nothing)
    cv2.createTrackbar('min_v', 'HSV_adjust', min_v, 255, nothing)
    cv2.createTrackbar('max_v', 'HSV_adjust', max_v, 255, nothing)

    while True:
        image_HSV = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2HSV)
        min_h = cv2.getTrackbarPos('min_h', 'HSV_adjust')
        max_h = cv2.getTrackbarPos('max_h', 'HSV_adjust') 
        min_s = cv2.getTrackbarPos('min_s', 'HSV_adjust') 
        max_s = cv2.getTrackbarPos('max_s', 'HSV_adjust') 
        min_v = cv2.getTrackbarPos('min_v', 'HSV_adjust') 
        max_v = cv2.getTrackbarPos('max_v', 'HSV_adjust') 

        lower_range = np.array([min_h, min_s, min_v])
        upper_range = np.array([max_h, max_s, max_v])

        mask = cv2.inRange(image_HSV, lower_range, upper_range)
        # 位运算
        licence_RGB = cv2.bitwise_and(image_RGB, image_RGB, mask=mask)

        cv2.imshow('licence_RGB', licence_RGB)

        if cv2.waitKey(1)&0xFF == 27:
            break


