#!/usr/bin/env python3
'''
ABOUT THE SCRIPT
It encodes the output from the RGB camera and both grayscale cameras
at the same time, having all encoder parameters set to maximum quality and FPS.
The RGB is set to 4K (3840x2160) and the grayscale are set to 1280x720 each, all at 25FPS.

INSTRUCTIONS
1. Run the script
2. Pressing Ctrl+C to stop the recording
3. Run:
    ffmpeg -framerate 25 -i [leftoutput].h264 -c copy [leftoutput].mp4
    ffmpeg -framerate 25 -i [rightoutput].h264 -c copy [rightoutput].mp4
    ffmpeg -framerate 25 -i [color].h265 -c copy [color].mp4
'''

import cv2
import depthai as dai
import os

from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
start_time = datetime.now()

# dd/mm/YY_H:M:S
date_time = now.strftime("%d-%m-%Y_%H:%M:%S")
print("Directory name: ", date_time)
if not os.path.exists(date_time):
    os.makedirs(date_time)

name1 = f'{date_time}/left.h264'
name2 = f'{date_time}/color.h265'
name3 = f'{date_time}/right.h264'

pipeline = dai.Pipeline()

# Nodes
colorCam = pipeline.createColorCamera()
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
monoCam = pipeline.createMonoCamera()
monoCam2 = pipeline.createMonoCamera()
ve1 = pipeline.createVideoEncoder()
ve2 = pipeline.createVideoEncoder()
ve3 = pipeline.createVideoEncoder()

ve1Out = pipeline.createXLinkOut()
ve2Out = pipeline.createXLinkOut()
ve3Out = pipeline.createXLinkOut()

# Properties
monoCam.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoCam2.setBoardSocket(dai.CameraBoardSocket.RIGHT)
ve1Out.setStreamName('ve1Out')
ve2Out.setStreamName('ve2Out')
ve3Out.setStreamName('ve3Out')

# Setting to 26fps will trigger error
ve1.setDefaultProfilePreset(1280, 720, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)
ve2.setDefaultProfilePreset(1920, 1080, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
ve3.setDefaultProfilePreset(1280, 720, 30, dai.VideoEncoderProperties.Profile.H264_MAIN)

# Link nodes
monoCam.out.link(ve1.input)
colorCam.video.link(ve2.input)
monoCam2.out.link(ve3.input)

ve1.bitstream.link(ve1Out.input)
ve2.bitstream.link(ve2Out.input)
ve3.bitstream.link(ve3Out.input)


# Pipeline is defined, now we can connect to the device
# device = dai.Device(pipeline)
# device.startPipeline()
with dai.Device(pipeline) as dev:

    # Prepare data queues
    outQ1 = dev.getOutputQueue('ve1Out', maxSize=30, blocking=True)
    outQ2 = dev.getOutputQueue('ve2Out', maxSize=30, blocking=True)
    outQ3 = dev.getOutputQueue('ve3Out', maxSize=30, blocking=True)

    # Start the pipeline
    dev.startPipeline()

    # Processing loop
    with open(name1, 'wb') as fileMono1H264, open(name2, 'wb') as fileColorH265, open(name3, 'wb') as fileMono2H264:
        while True:
            try:
                time_elapsed = datetime.now() - start_time
                print('\r','Press Ctrl+C to stop encoding. Recording (hh:mm:ss.ms) {}'.format(time_elapsed), end='')
                # Empty each queue
                while outQ1.has():
                    outQ1.get().getData().tofile(fileMono1H264)

                while outQ2.has():
                    outQ2.get().getData().tofile(fileColorH265)

                while outQ3.has():
                    outQ3.get().getData().tofile(fileMono2H264)

                if time_elapsed.total_seconds() > 60:
                    break
            except KeyboardInterrupt:
                break

    print("To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
    cmd = "ffmpeg -framerate 25 -i {} -c copy {}"
    print(cmd.format(name1, name1[:-4] + "mp4"))
    print(cmd.format(name3, name3[:-4] + "mp4"))
    print(cmd.format(name2, name2[:-4] + "mp4"))
