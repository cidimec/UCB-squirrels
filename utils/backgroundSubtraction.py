import cv2
import numpy as np
from utils import LDA
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

class bsub:
    def __init__(self):
        self.silhouettes = []
        self.num_sil = 0
        self.bg = None
        self.margin = 10
        self.classID = 0
        self.pred_list = []
        # self.model = tf.keras.models.load_model('../models/bestDB/128x128_acc_0.9471_loss_0.0605_val-acc_0.9476_val-loss_0.0586_0.22M_29-07-21-DB-best')
        # self.model = tf.keras.models.load_model('../models/UCB300/128x128unet_acc:0.9540_loss:0.0590_val-acc:0.9538_val-loss:0.0594_0.22M_01-08-21-DB_UCB300_E:10x1E-4:5x1E-5')
        self.model = tf.keras.models.load_model('../models/BioSmart/128x128unet_acc_0.9451_loss_0.0845_val-acc_0.9457_val-loss_0.0826_0.22M_24-10-21-DB_UCB300_Epochs_10x1E-4_5x1E-5')
        self.model = tf.keras.models.load_model('../models/BioSmart/128x128unet_acc_0.9525_loss_0.0641_val-acc_0.9522_val-loss_0.0649_0.59M_24-10-21-DB_UCB300_Epochs_10x1E-4_5x1E-5')

        # self.model = tf.keras.models.load_model('../models/UCB300/128x128_acc_0.9549_loss_0.0886_val-acc_0.9554_val-loss_0.0873_0.22M_01-08-21-DB_UCB300_E_10x1E-4_5x1E-5')

    def setBackound(self, frame):
        self.bg = frame
        self.height, self.width, _ = frame.shape

    def clear(self):
        self.silhouettes = []
        self.pred_list = []
        self.num_sil = 0
        # self.bg = None

    def fine_mask(self, mask):
        ''' Takes a raw mask as input and returns the biggest contour mask '''
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key=cv2.contourArea)
        out = np.zeros_like(mask)
        out = cv2.drawContours(out, [c], 0, 255, -1)
        return out

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
            # roi_mask = self.fine_mask(roi_mask)
        # print(roi_mask.shape, roi_mask.dtype, type(roi_mask))
        self.silhouettes.append(roi_mask)
        # path = os.path.join(self.dir, str(self.num_sil))
        # cv2.imwrite(str(self.num_sil)+ '.png', roi_mask)
        self.num_sil +=1
        if self.num_sil % 15 == 0:
            GEI, _ = LDA.GEI_generator(self.silhouettes)
            # cv2.imwrite(f'{self.num_sil}.png', GEI)
            self.classID = int(LDA.inference(GEI)[0])
            self.pred_list.append(self.classID)
            print(self.classID)
        return roi_mask

    def inference(self, img, width=128, height=128):
        h, w, _ = img.shape
        # print(img.shape)
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
        col2 = col2 + 2*margin if col2 + 2*margin <= self.width else self.width
        row2 = row2 + 2*margin if row2 + 2*margin <= self.height else self.height
        return (col1, row1, col2, row2)
