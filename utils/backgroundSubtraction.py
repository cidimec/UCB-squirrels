import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import LDA


class bsub:
    def __init__(self):
        self.silhouettes = []
        self.num_sil = 0
        self.bg = None
        self.dir = 'test'
        self.margin = 30
        self.classID = 0

    def setBackound(self, frame):
        self.bg = frame
        self.height, self.width, _ = frame.shape

    def substract(self, frame, bbox):
        diff = cv2.subtract(self.bg, frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_gray = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

        col1, row1, col2, row2 = self.limit_bbox(bbox)
        roi = diff_gray[row1: row2, col1: col2]
        self.silhouettes.append(roi)
        # path = os.path.join(self.dir, str(self.num_sil))
        # cv2.imwrite(path+ '.png', roi)
        self.num_sil +=1
        if self.num_sil % 10 == 0:
            GEI, _ = LDA.GEI_generator(self.silhouettes)
            self.classID = LDA.inference(GEI)
            print(self.classID)
        return diff_gray, roi

    def limit_bbox(self, raw_bbox):
        col1, row1, col2, row2 = raw_bbox
        margin = self.margin
        col1 = col1 - margin if col1 - margin >= 0 else 0
        row1 = row1 - margin if row1 - margin >= 0 else 0
        col2 = col2 + margin if col2 + margin <= self.width else self.width
        row2 = row2 + margin if row2 + margin <= self.height else self.height
        return (col1, row1, col2, row2)
