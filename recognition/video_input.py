#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import os
import depthai as dai
import numpy as np
from time import monotonic

sys.path.append(os.path.abspath('../'))
from utils.backgroundSubtraction import bsub
# sys.path.append(os.path.abspath('../recognition'))

nnPath = '../models/mobilenet-ssd_openvino_2021.2_8shave.blob'
videoPath = '../data/videos/001-nm-01-090.avi'
videoPath = '/home/israel/Downloads/CASIA/DatasetB-1/video/001-nm-01-090.avi'
id_label = int(videoPath.split('/')[-1].split('-')[0])
bsub = bsub()

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

# Define sources and outputs
nn = pipeline.createMobileNetDetectionNetwork()

xinFrame = pipeline.createXLinkIn()
xinFrame.setStreamName("inFrame")

nnOut = pipeline.createXLinkOut()
nnOut.setStreamName("nn")

# Properties
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Linking
xinFrame.out.link(nn.input)
nn.out.link(nnOut.input)

cv2.namedWindow('rgb')
cv2.moveWindow('rgb', 0, 0)
cv2.namedWindow('background')
cv2.moveWindow('background', 0, 500)
cv2.namedWindow('roi')
cv2.moveWindow('roi', 0, 700)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    startTime = monotonic()
    nn_counter = 0
    frames_counter = 0
    # Input queue will be used to send video frames to the device.
    qIn = device.getInputQueue(name="inFrame")
    # Output queue will be used to get nn data from the video frames.
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    def displayFrame(name, frame):
        current = frame.copy()
        bbox = None
        roi = None
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            # cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_ITALIC, 0.5, 255)
            # cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_ITALIC, 0.5, 255)
            label = 'ID: {}, PRED: {}'.format(id_label, int(bsub.classID))
            cv2.putText(frame, label, (bbox[0], bbox[1] - 7), cv2.FONT_ITALIC, 0.5, 255)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(frame, "Cam fps: {:.2f}".format(frames_counter / (monotonic() - startTime)),
                    (2, frame.shape[0] - 15), cv2.FONT_ITALIC, 0.4, (255, 255, 255))
        cv2.putText(frame, "NN fps: {:.2f}".format(nn_counter / (monotonic() - startTime)),
                    (2, frame.shape[0] - 4), cv2.FONT_ITALIC, 0.4, (255, 255, 255))
        if bsub.bg is None:
            bsub.setBackound(current)
            background = np.zeros_like(current)
        elif bbox is not None:
            background, roi = bsub.substract(current, bbox)
        # Show the frame
        cv2.imshow(name, frame)
        if roi is not None:
            cv2.imshow('background', background)
            cv2.imshow('roi', roi)

    cap = cv2.VideoCapture(videoPath)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    delay = int(1000 / fps)
    while cap.isOpened():
        read_correctly, frame = cap.read()
        if not read_correctly:
            break
        else:
            frames_counter +=1

        img = dai.ImgFrame()
        img.setData(to_planar(frame, (300, 300)))
        img.setTimestamp(monotonic())
        img.setWidth(300)
        img.setHeight(300)
        qIn.send(img)

        inDet = qDet.get()

        if inDet is not None:
            detections = inDet.detections
            nn_counter +=1

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(40) == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
