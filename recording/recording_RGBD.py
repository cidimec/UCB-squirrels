#!/usr/bin/env python3
'''
ABOUT THE SCRIPT
It encodes the output from the RGB camera and both grayscale cameras
at the same time, having all encoder parameters set to maximum quality and FPS.
The RGB is set to 4K (3840x2160) and the grayscale are set to 1280x720 each, all at 25FPS.

INSTRUCTIONS
1. Run the script
2. Pressing Ctrl+C to stop the recording
3. Run the lines in the convert.txt file created along the video enconded files
'''

import cv2
import depthai as dai
import os
import utils
from datetime import datetime
import argparse

# Retrieve the args
parser = argparse.ArgumentParser(description='Parameters available to set')
parser.add_argument("-dir", "--directory", default=None, help="Directory where the videos are stored")
parser.add_argument("-rgb", "--c_res", type=int, default=1080, help="RGB camera resolution height: (1920x)1080, (3840x)2160 or (4056x)3040. Default: 1080 ")
parser.add_argument("-mono", "--m_res", type=int, default=720, help="Mono cam resolution height: (1280x)720, (1280x)800 or (640x)400. Default: 720")
parser.add_argument("-f", "--fps", type=int, default=30, action='store', help="Defines the output frame rate. Default: 30")
parser.add_argument("-sec", "--sec", type=int, default=30, action='store', help="Defines the time (seconds) that the videos will be recorded. Default: 30")
parser.add_argument("-id", "--subject", type=str, default=None, help="Defines the subject's name (Optinal)")
parser.add_argument("-w", "--walk", type=str, default=None, help="Defines the walking type (Optional)")

parser.add_argument("-v", "--verb", type=int, default=0, help="Defines the output verbosity")
args = parser.parse_args()

# Configure the parameters
fps = args.fps
verbose = True
if args.c_res == 2160:
    color_resolution = dai.ColorCameraProperties.SensorResolution.THE_4_K
    color_shape = (3840, 2160)
elif args.c_res == 3040:
    color_resolution = dai.ColorCameraProperties.SensorResolution.THE_12_MP
    color_shape = (4056, 3040)
else:
    color_resolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
    color_shape = (1920, 1080)

if args.m_res == 720:
    mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
    mono_shape = (1280, 720)
elif args.m_res == 800:
    mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_800_P
    mono_shape = (1280, 800)
else:
    mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
    mono_shape = (640, 400)

# Create a directory with the current time
if args.directory is None:
    name = utils.current()
    if args.subject is not None:
        name = f'{name[:-1]}_id:{args.subject}_walk:{args.walk}/'
    directory = utils.checkFileExist(name, create=True)
    dir_name = directory.split('/')[-2]

else:
    directory = utils.checkFileExist(args.directory, create=True)
    dir_name = directory.split('/')[-1]


print(f'Configuration-> Folder:{dir_name} Color resolution: {color_shape}, Mono resolution: {mono_shape}, FPS: {fps}')

# date_time = now.strftime("%d-%m-%Y_%H:%M:%S")
# print("Directory name: ", date_time)
# if not os.path.exists(date_time):
#     os.makedirs(date_time)

name1 = os.path.join(dir_name, 'left_video.h264')
name2 = os.path.join(dir_name, 'color_video.h265')
name3 = os.path.join(dir_name, 'right_video.h264')

pipeline = dai.Pipeline()

# Nodes
colorCam = pipeline.createColorCamera()
monoCam = pipeline.createMonoCamera()
monoCam2 = pipeline.createMonoCamera()

# Properties
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
monoCam.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoCam2.setBoardSocket(dai.CameraBoardSocket.RIGHT)

colorCam.setResolution(color_resolution)
monoCam.setResolution(mono_resolution)
monoCam2.setResolution(mono_resolution)

colorCam.setFps(fps)
monoCam.setFps(fps)
monoCam2.setFps(fps)

ve1 = pipeline.createVideoEncoder()
ve2 = pipeline.createVideoEncoder()
ve3 = pipeline.createVideoEncoder()

ve1Out = pipeline.createXLinkOut()
ve2Out = pipeline.createXLinkOut()
ve3Out = pipeline.createXLinkOut()

ve1Out.setStreamName('ve1Out')
ve2Out.setStreamName('ve2Out')
ve3Out.setStreamName('ve3Out')

# Setting to 26fps will trigger error
ve1.setDefaultProfilePreset(mono_shape[0], mono_shape[1], fps, dai.VideoEncoderProperties.Profile.H264_MAIN)
ve2.setDefaultProfilePreset(color_shape[0], color_shape[1], fps, dai.VideoEncoderProperties.Profile.H265_MAIN)
ve3.setDefaultProfilePreset(mono_shape[0], mono_shape[1], fps, dai.VideoEncoderProperties.Profile.H264_MAIN)

# Link nodes
monoCam.out.link(ve1.input)
colorCam.video.link(ve2.input)
monoCam2.out.link(ve3.input)

ve1.bitstream.link(ve1Out.input)
ve2.bitstream.link(ve2Out.input)
ve3.bitstream.link(ve3Out.input)

# Pipeline is defined, now we can connect to the device
device = dai.Device(pipeline)

# Prepare data queues
outQ1 = device.getOutputQueue('ve1Out', maxSize=30, blocking=True)
outQ2 = device.getOutputQueue('ve2Out', maxSize=30, blocking=True)
outQ3 = device.getOutputQueue('ve3Out', maxSize=30, blocking=True)

# Processing loop
with open(name1, 'wb') as fileMono1H264, open(name2, 'wb') as fileColorH265, open(name3, 'wb') as fileMono2H264:
    start_time = datetime.now()
    while True:
        try:
            time_elapsed = datetime.now() - start_time
            print('\r', 'Press Ctrl+C to stop encoding. Recording (hh:mm:ss.ms) {}'.format(time_elapsed), end='')
            # Empty each queue
            while outQ1.has():
                outQ1.get().getData().tofile(fileMono1H264)
            while outQ2.has():
                outQ2.get().getData().tofile(fileColorH265)
            while outQ3.has():
                outQ3.get().getData().tofile(fileMono2H264)
            # if time_elapsed.total_seconds() > args.sec:
            #     print('Stopping recording due to time limit')
            #     break
        except KeyboardInterrupt:
            break

print("To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands in the .txt file")
f = open(os.path.join(dir_name, "convert.txt"),"w+")
cmd = "ffmpeg -framerate {} -i {} -c copy {} \n\n"
f.write(cmd.format(fps, name1, name1[:-4] + "mp4"))
f.write(cmd.format(fps, name3, name3[:-4] + "mp4"))
f.write(cmd.format(fps, name2, name2[:-4] + "mp4"))
