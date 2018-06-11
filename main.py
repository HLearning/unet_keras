#-*- coding:utf-8 -*-

from data_extension import DataExtension
from unet import UNet
import cv2
if __name__ == '__main__':

    de = DataExtension()
    # 扩展数据
    de.data_extension()
    # 创建训练集
    de.create_train_data()
    # 加载训练集
    train_img, train_lbl = de.load_train_data()

    unet = UNet()
    #unet.unet_train(train_img, train_lbl)
    img = cv2.imread('./data/test/test.png')
    label = unet.unet_predict_img(img)
    cv2.imwrite('test_label.png', label)