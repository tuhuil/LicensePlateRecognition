import cv2

'''
 图像预处理
读取图片，得到指定尺寸的彩色图和灰度图
'''
def PictureDispose(picture_path):
    # 读取图片，得到指定尺寸的彩色图和灰度图
    image_color = cv2.imread(picture_path, cv2.IMREAD_COLOR) # 读取图片,彩色图
    image_color = cv2.resize(image_color, (600, 400)) # 修改图片大小
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY) # 转换为灰度图

    # cv2.imshow('1', image_gray )

    # 滤波
    image_gray = cv2.bilateralFilter(image_gray, 13, 15, 15)

    # cv2.imshow('2', image_gray )

    return image_color, image_gray

'''
 车牌定位
'''
def License_find(image_color, image_gray):
    image_edged = cv2.Canny(image_gray, 30, 200) # 边缘检测
    cv2.imshow('image_edged', image_edged )


if __name__ == "__main__":
    # 1.图像预处理
    image_color, image_gray = PictureDispose('picture/1.jpg')

    # 车牌定位
    License_find(image_color, image_gray)

    # cv2.imshow('image_color', image_color )
    # cv2.imshow('image_gray', image_gray )



    cv2.waitKey(0)
    cv2.destroyAllWindows()