from data import conv_image, train_generator
from unet import UNet
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
if __name__ == '__main__':
    # tif to png
    conv_image()

    unet = UNet()
    is_train = False
    if is_train:

        model = unet.vgg10_unet(input_shape=(256,256,3), weights='imagenet')
        for index in range(15):
            model.layers[index].trainable = False

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model_checkpoint = ModelCheckpoint('unet.h5', monitor='loss', verbose=1, save_best_only=True)
        model.fit_generator(train_generator(batch_size=8),
                            steps_per_epoch=100,
                            epochs=50,
                            validation_data=train_generator(batch_size=8),
                            validation_steps=50,
                            callbacks=[model_checkpoint])

    else:
        model = unet.vgg10_unet(input_shape=(256,256,3))
        model.load_weights('unet.h5')

        x = cv2.imread('./data/test/images/imgs/0.png')
        x = cv2.resize(x, (256,256))
        x = x / 255.0
        x = np.array([x])
        print(x.shape)
        mask = model.predict(x, batch_size=None, verbose=0, steps=None)
        mask = mask[0]
        print(mask.shape)
        mask = np.argmax(mask, axis=2)
        mask = mask * 255
        print(mask.shape)
        mask = mask.reshape((256,256,1))
        print(mask.shape)
        cv2.imwrite('mask.png', mask)
