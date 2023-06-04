import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 分割出车牌区域
def LicenseSegmentation(image_RGB):
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


    '''
    2.分割出车牌区域
    '''
    image_RGB = image_origin 
    image_license = LicenseSegmentation(image_RGB) # 车牌区域

    # 显示原始图像
    cv2.imshow('origin', image_origin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

