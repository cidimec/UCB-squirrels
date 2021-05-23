#!/usr/bin/env python3

import cv2
import depthai as dai
import os

from datetime import datetime

# datetime object containing current date and time
now = datetime.now()

# dd/mm/YY_H:M:S
date_time = now.strftime("%d-%m-%Y_%H:%M:%S")
print("Directory name: ", date_time)
if not os.path.exists(date_time):
    os.makedirs(date_time)

name1 = f'{date_time}/left_{date_time}.mp4'
name2 = f'{date_time}/color_{date_time}.mp4'
name3 = f'{date_time}/right_{date_time}.mp4'

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.createColorCamera()
# camRgb.setPreviewSize(, 500)
# camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setInterleaved(False)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)


# Define a source - two mono (grayscale) cameras
camLeft = pipeline.createMonoCamera()
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

camRight = pipeline.createMonoCamera()
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Get data
print(camRgb.getVideoSize(), camRgb.getFps())
print(camLeft.getResolutionSize(), camLeft.getFps())
print(camRight.getResolutionSize(), camRight.getFps())

# Create output
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

# Create outputs
xoutLeft = pipeline.createXLinkOut()
xoutLeft.setStreamName('left')
camLeft.out.link(xoutLeft.input)

xoutRight = pipeline.createXLinkOut()
xoutRight.setStreamName('right')
camRight.out.link(xoutRight.input)

# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
outleft = cv2.VideoWriter(name1, cv2.VideoWriter_fourcc(*'MPEG'), 30, camLeft.getResolutionSize(), 0)
outcolor = cv2.VideoWriter(name2, cv2.VideoWriter_fourcc(*'MPEG'), 30, camRgb.getVideoSize())
outright = cv2.VideoWriter(name3, cv2.VideoWriter_fourcc(*'MPEG'), 30, camRight.getResolutionSize(), 0)

# Pipeline is defined, now we can connect to the device
# with dai.Device(pipeline) as device:
    # Start pipeline
    # device.startPipeline()
device = dai.Device(pipeline)
device.startPipeline()

# Output queue will be used to get the rgb frames from the output defined above
qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)
qLeft = device.getOutputQueue(name="left", maxSize=30, blocking=False)
qRight = device.getOutputQueue(name="right", maxSize=30, blocking=False)

frameRgb = None
frameLeft = None
frameRight = None

while True:
    try:
        inRgb = qRgb.tryGet()
        # print(qRgb.get().getData().shape)
        inLeft = qLeft.tryGet()
        inRight = qRight.tryGet()
        if inRgb is not None:
            frameRgb = inRgb.getCvFrame()
            # frameRgb = bytearray(inRgb.getData())

        if inLeft is not None:
            frameLeft = inLeft.getCvFrame()

        if inRight is not None:
            frameRight = inRight.getCvFrame()

        # Retrieve 'bgr' (opencv format) frame
        # show the frames if available
        if frameRgb is not None:
            # cv2.imshow("bgr", frameRgb)
            outcolor.write(frameRgb)

        if frameLeft is not None:
            # cv2.imshow("left", frameLeft)
            outleft.write(frameLeft)

        if frameRight is not None:
            # cv2.imshow("right", frameRight)
            outright.write(frameRight)

        if cv2.waitKey(1) == ord('q'):
            break

    except KeyboardInterrupt:
        outcolor.release()
        outleft.release()
        outright.release()
        break
