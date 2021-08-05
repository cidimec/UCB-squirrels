import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import LDA
import tensorflow as tf

class bsub:
    def __init__(self):
        self.silhouettes = []
        self.num_sil = 0
        self.bg = None
        self.dir = 'test'
        self.margin = 30
        self.classID = 0

        self.model = tf.keras.models.load_model('../models/bestDB/128x128_acc_0.9471_loss_0.0605_val-acc_0.9476_val-loss_0.0586_0.22M_29-07-21-DB-best')

    def setBackound(self, frame):
        self.bg = frame
        self.height, self.width, _ = frame.shape

    def substract(self, frame, bbox, mode='naive'):
        if mode == 'naive':
            diff = cv2.subtract(self.bg, frame)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, diff_gray = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

            col1, row1, col2, row2 = self.limit_bbox(bbox)
            roi_mask = diff_gray[row1: row2, col1: col2]
        elif mode == 'deep':
            col1, row1, col2, row2 = self.limit_bbox(bbox)
            roi = frame[row1: row2, col1: col2]
            roi_mask = self.inference(roi)
        # print(roi_mask.shape, roi_mask.dtype, type(roi_mask))
        self.silhouettes.append(roi_mask)
        # path = os.path.join(self.dir, str(self.num_sil))
        # cv2.imwrite(path+ '.png', roi)
        self.num_sil +=1
        if self.num_sil % 10 == 0:
            GEI, _ = LDA.GEI_generator(self.silhouettes)
            self.classID = LDA.inference(GEI)
            print(self.classID)
        return roi_mask

    def inference(self, img, width=128, height=128):
        h, w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tf_image = tf.image.resize(tf.convert_to_tensor(np.array(img)), (width, height))/ 255
        output = self.model.predict(tf.expand_dims(tf_image, 0))
        output = tf.image.resize(output, (h, w))
    #     output = model.predict(tf.stack([tf_image, tf_image, tf_image, tf_image, tf_image]))
        out = np.where(output[0, :, :, 0]>0.4, 1, 0)
        # out = cv2.resize(out, (w, h), cv2.INTER_LINEAR)
        return (out * 255).astype('uint8')

    def limit_bbox(self, raw_bbox):
        col1, row1, col2, row2 = raw_bbox
        margin = self.margin
        col1 = col1 - margin if col1 - margin >= 0 else 0
        row1 = row1 - margin if row1 - margin >= 0 else 0
        col2 = col2 + margin if col2 + margin <= self.width else self.width
        row2 = row2 + margin if row2 + margin <= self.height else self.height
        return (col1, row1, col2, row2)
