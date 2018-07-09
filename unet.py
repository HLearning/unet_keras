import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint
import cv2
from keras.applications.vgg16 import VGG16


# 定义Unet网络
class UNet(object):
    def __init__(self, img_h=256, img_w=256):
        self.img_h = img_h
        self.img_w = img_w


    def vgg10_unet(self, input_shape=(256,256,3), weights='imagenet'):
        vgg16_model = VGG16(input_shape=input_shape, weights=weights, include_top=False)

        block4_pool = vgg16_model.get_layer('block4_pool').output
        block5_conv1 =  Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block4_pool)
        block5_conv2 =  Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block5_conv1)
        block5_drop = Dropout(0.5)(block5_conv2)

        block6_up = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(block5_drop))
        block6_merge = Concatenate(axis=3)([vgg16_model.get_layer('block4_conv3').output, block6_up])
        block6_conv1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_merge)
        block6_conv2 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_conv1)
        block6_conv3 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block6_conv2)


        block7_up = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(block6_conv3))
        block7_merge = Concatenate(axis=3)([vgg16_model.get_layer('block3_conv3').output, block7_up])
        block7_conv1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_merge)
        block7_conv2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv1)
        block7_conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv2)


        block8_up = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(block7_conv3))
        block8_merge = Concatenate(axis=3)([vgg16_model.get_layer('block2_conv2').output, block8_up])
        block8_conv1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block8_merge)
        block8_conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block8_conv1)

        block9_up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(block8_conv2))
        block9_merge = Concatenate(axis=3)([vgg16_model.get_layer('block1_conv2').output, block9_up])
        block9_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_merge)
        block9_conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv1)

        block10_conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv2)
        block10_conv2 = Conv2D(2, 1, activation='sigmoid')(block10_conv1)

        model = Model(inputs=vgg16_model.input, outputs=block10_conv2)
        return model