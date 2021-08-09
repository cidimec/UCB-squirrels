#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
from time import monotonic

# Get argument first
nnPath = '../models/64x64unet6shaves.blob'
nnPath = '/home/israel/Downloads/frozen_graph.blob'
nnPath = '../models/unet6shaves.blob'

videoPath = '/home/israel/repos/UCB-squirrels/data/videos/001-bg-01-090.avi'
# videoPath = '/home/israel/Downloads/UCB/clips/090/001/nm/nm-01.avi'
# Create pipeline
pipeline = dai.Pipeline()

xinFrame = pipeline.createXLinkIn()
nnOut = pipeline.createXLinkOut()

xinFrame.setStreamName("inFrame")
nnOut.setStreamName("nn")

# Properties
# Define sources and outputs
nn = pipeline.createNeuralNetwork()
# nn.setConfidenceThreshold(0.5)
nn.setBlobPath(nnPath)
# nn.setNumInferenceThreads(2)
# nn.input.setBlocking(False)

# Linking
xinFrame.out.link(nn.input)
nn.out.link(nnOut.input)

# Connect to device and start pipeline
device = dai.Device(pipeline)

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
    # cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    cv2.putText(frame, f"Fps: {fps}", (10, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
    # Show the frame
    cv2.imshow(name, frame)
    # cv2.imshow('out', mask)


size = 128
start_time = monotonic()
counter = 0
fps = 0
cap = cv2.VideoCapture(videoPath)
while cap.isOpened():
    read_correctly, frame = cap.read()
    if not read_correctly:
        break

    img = dai.ImgFrame()
    img.setData(to_planar(frame, (size, size)))
    img.setTimestamp(monotonic())
    img.setWidth(size)
    img.setHeight(size)
    qIn.send(img)

    inDet = qDet.get()
    print(inDet.getAllLayerNames())
    out = inDet.getLayerFp16("model_7/conv2d_transpose_36/Sigmoid")
    output = np.array(out).reshape(size, size)
    norm_image = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # norm_image = (norm_image*255).astype('uint8')
    # print(norm_image.dtype)
    # print(norm_image.max(), norm_image.min())
    norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB)
    ret, mask = cv2.threshold(norm_image, 70, 255, cv2.THRESH_BINARY)
    print(norm_image.shape)
    cv2.imshow('out', mask)
    counter+=1
    if (monotonic() - start_time) > 1:
        fps = counter // (monotonic() - start_time)
        counter = 0
        start_time = monotonic()
    # if inDet is not None:
    #     detections = inDet.detections
    #
    if frame is not None:
        displayFrame("rgb", frame)

    if cv2.waitKey(30) == ord('q'):
        break
