#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
import utils
import os
# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(720, 480)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

# Create output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

q_rgb_list = []
paths_list = []
n_frame = 0

# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    for device_info in dai.Device.getAllAvailableDevices():
        device = stack.enter_context(dai.Device(pipeline, device_info))
        id = device_info.getMxId()
        dir_path = utils.checkFileExist(id, True)
        paths_list.append(dir_path)
        print("Conected to " + id)
        device.startPipeline()

        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=10, blocking=False)
        q_rgb_list.append(q_rgb)

    while True:
        for i, (q_rgb, dir_path) in enumerate(zip(q_rgb_list, paths_list)):
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                name = 'rgb-' + str(n_frame + 1).zfill(5) + '.png'
                path = os.path.join(dir_path, name)
                frame = in_rgb.getCvFrame()
                cv2.imshow("rgb-" + str(i + 1), frame)
                cv2.imwrite(path, frame)
                n_frame +=1

        if cv2.waitKey(1) == ord('q'):
            break
