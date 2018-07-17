import numpy as np
from keras.models import *
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import cv2

def train_generator(batch_size=32):
    data_gen_args = dict(featurewise_center=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         fill_mode="constant",
                         cval=255,
                         horizontal_flip=True,
                         vertical_flip=True,
                         zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    image_generator = image_datagen.flow_from_directory(
        'data/train/images',
        class_mode=None,
        batch_size=batch_size,
        color_mode='rgb',
        target_size=(512,512),
        #save_to_dir='./data/gen/images',
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        'data/train/masks',
        class_mode=None,
        color_mode='grayscale',
        target_size=(512,512),
        batch_size=batch_size,
        #save_to_dir='./data/gen/masks',
        seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    for (imgs, masks) in train_generator:
        imgs = imgs / 255.0
        masks = masks / 255.0
        yield (imgs,masks)


def vgg10_unet(input_shape=(256,256,3), weights='imagenet'):
    vgg16_model = VGG16(input_shape=input_shape, weights=weights, include_top=False)

    block4_pool = vgg16_model.get_layer('block4_pool').output
    block5_conv1 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block4_pool)
    block5_conv2 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(block5_conv1)
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
    block10_conv2 = Conv2D(1, 1, activation='sigmoid')(block10_conv1)

    model = Model(inputs=vgg16_model.input, outputs=block10_conv2)
    return model


if __name__ == '__main__':
    is_train = False
    if is_train:
        model = vgg10_unet(input_shape=(512,512,3), weights='imagenet')

        for index in range(15):
            model.layers[index].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model_checkpoint = ModelCheckpoint('unet.h5', monitor='loss', verbose=1, save_best_only=True)
        model.fit_generator(train_generator(batch_size=4),
                            steps_per_epoch=200,
                            epochs=50,
                            validation_data=train_generator(batch_size=4),
                            validation_steps=50,
                            callbacks=[model_checkpoint])

    else:
        model = vgg10_unet(input_shape=(512,512,3))
        model.load_weights('unet.h5')

        for i in range(30):
            x = cv2.imread('./data/test/images/imgs/%d.png'%i)
            x = x / 255.0
            x = np.array([x])
            mask = model.predict(x, batch_size=None, verbose=0, steps=None)
            mask = mask[0]
            mask = mask * 255
            cv2.imwrite('./data/test/masks/imgs/%d.png'%i, mask)