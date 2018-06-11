#-*- coding:utf-8 -*-

import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import cv2
import keras.backend.tensorflow_backend as KTF


# 定义Unet网络
class UNet(object):
    def __init__(self, img_h=128, img_w=128):
        self.img_h = img_h
        self.img_w = img_w

    def unet_net(self):
        inputs = Input((self.img_h, self.img_w, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = Concatenate(axis=3)([conv4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = Concatenate(axis=3)([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(3, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def unet_train(self, train_image, train_label):
        print("train...")
        model = self.unet_net()
        '''
        如果验证损失下降， 那么在每个训练轮之后保存模型。
        '''
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(train_image, train_label, batch_size=8, epochs=2, verbose=1,
                  shuffle=True, callbacks=[model_checkpoint])


    def unet_predict_img(self, img):
        model = self.unet_net()
        model.load_weights('./unet.hdf5')
        imgdatas = []
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        imgdatas.append(img)
        imgdatas = np.array(imgdatas)
        imgdatas = imgdatas.astype('float32')
        imgdatas = imgdatas / 255.0
        test_label = model.predict(imgdatas, batch_size=1, verbose=0)
        label = test_label[0]
        label[label>0.5] = 1
        label[label<=0.5] = 0
        label = label * 255
        label = label.astype('uint8')
        #print(label.shape)
        return label
