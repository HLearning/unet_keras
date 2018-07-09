from libtiff import TIFF
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

def tiff_to_image_array(tiff_image_name, out_folder, out_type):
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    tif = TIFF.open(tiff_image_name, mode = "r")
    idx = 0
    img_list = list(tif.iter_images())
    for img in img_list:
        im_name = out_folder + str(idx) + out_type
        cv2.imwrite(im_name, img)
        idx = idx + 1
# tiff to png
def conv_image():
    tiff_to_image_array('data/origin/train-volume.tif', './data/train/images/imgs/', '.png')
    tiff_to_image_array('data/origin/train-labels.tif', './data/train/masks/imgs/', '.png')
    tiff_to_image_array('data/origin/test-volume.tif', './data/test/images/imgs/', '.png')


def train_generator(batch_size=32):
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.5)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1

    image_generator = image_datagen.flow_from_directory(
        'data/train/images',
        class_mode=None,
        batch_size=batch_size,
        color_mode='rgb',
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        'data/train/masks',
        class_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    for (imgs, masks) in train_generator:


        imgs = imgs / 255.0
        masks = masks / 255.0
        masks = to_categorical(masks, num_classes=2)
        yield (imgs,masks)

