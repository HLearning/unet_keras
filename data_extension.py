import cv2
import os
import numpy as np
import glob

class DataExtension(object):

    def __init__(self, imgH=128, imgW=128):
        self.imgH = imgH
        self.imgW = imgW
        self.save_path = 'data/train'


    # 数据扩展
    def data_extension(self):
        imagePath = 'data/train/train_image.png'
        labelPath = 'data/train/train_label.png'

        img = cv2.imread(imagePath)
        lbl = cv2.imread(labelPath)

        print(img.shape)
        print(img.shape)

        img = cv2.resize(img, (640, 160))
        lbl = cv2.resize(lbl, (640, 160))

        i = 0
        cv2.imwrite(self.save_path + "/image/%d.png" % (i), img)
        cv2.imwrite(self.save_path + "/label/%d.png" % (i), lbl)
        i = i + 1
        cv2.imwrite(self.save_path + "/image/%d.png" % (i), cv2.flip(img, 1))
        cv2.imwrite(self.save_path + "/label/%d.png" % (i), cv2.flip(lbl, 1))
        i = i + 1
        cv2.imwrite(self.save_path + "/image/%d.png" % (i), cv2.flip(img, 0))
        cv2.imwrite(self.save_path + "/label/%d.png" % (i), cv2.flip(lbl, 0))
        i = i + 1
        cv2.imwrite(self.save_path + "/image/%d.png" % (i), cv2.flip(img, -1))
        cv2.imwrite(self.save_path + "/label/%d.png" % (i), cv2.flip(lbl, -1))
        i = i + 1

        for j in range(4):
            for k in range(64):
                # 偶数值， 方便补零操作
                stepH = 8
                stepW = 8

                # 这个图片是一个子集， 将子集变形
                image = img[0 + j * stepH:self.imgH + (j - 2) * stepH, 0 + k * stepW:self.imgW + (k - 2) * stepW]
                # 这个图片是一个子集， 将子集变形
                label = lbl[0 + j * stepH:self.imgH + (j - 2) * stepH, 0 + k * stepW:self.imgW + (k - 2) * stepW]

                cv2.imwrite(self.save_path + "/image/%d.png" % (i), image)
                cv2.imwrite(self.save_path + "/label/%d.png" % (i), label)
                i = i + 1
                cv2.imwrite(self.save_path + "/image/%d.png" % (i), cv2.flip(image, 1))
                cv2.imwrite(self.save_path + "/label/%d.png" % (i), cv2.flip(label, 1))
                i = i + 1
                cv2.imwrite(self.save_path + "/image/%d.png" % (i), cv2.flip(image, 0))
                cv2.imwrite(self.save_path + "/label/%d.png" % (i), cv2.flip(label, 0))
                i = i + 1
                cv2.imwrite(self.save_path + "/image/%d.png" % (i), cv2.flip(image, -1))
                cv2.imwrite(self.save_path + "/label/%d.png" % (i), cv2.flip(label, -1))
                i = i + 1

    # 创建训练数据和验证数据
    def create_train_data(self):
        i = 0
        imgs = glob.glob(self.save_path + "/image/*.png")
        print(len(imgs))
        images = np.ndarray((len(imgs), self.imgH, self.imgW, 3), dtype=np.uint8)
        labels = np.ndarray((len(imgs), self.imgH, self.imgW, 3), dtype=np.uint8)
        imgsNumber = len(imgs)

        for i in range(imgsNumber):
            imgPath = self.save_path + "/image/" + str(i) + ".png"
            lblPath = self.save_path + "/label/" + str(i) + ".png"
            img = cv2.imread(imgPath)
            lbl = cv2.imread(lblPath)
            img = cv2.resize(img, (128, 128))
            lbl = cv2.resize(lbl, (128, 128))

            images[i] = img
            labels[i] = lbl

        np.save(self.save_path + '/image.npy', images)
        np.save(self.save_path + '/label.npy', labels)
        print('Saving to .npy files done.')

    # 加载训练
    def load_train_data(self):
        print('load train images...')
        img = np.load(self.save_path + "/image.npy")
        lbl = np.load(self.save_path + "/label.npy")

        img = img.astype('float32')
        lbl = lbl.astype('float32')

        img /= 255.0
        lbl /= 255.0
        return img, lbl

