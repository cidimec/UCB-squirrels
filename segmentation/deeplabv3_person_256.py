#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import sys

'''
RUNNING INSTRUCTIONS
Video ->python deeplabv3_person_256.py -cam {PATH_OF_VIDEO}
Camera -> python deeplabv3_person_256.py
'''

cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb')
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='../models/deeplab_v3_plus_mvn2_decoder_256_openvino_2021.2_6shave.blob', type=str)
# parser.add_argument("-vid", "--vid_input", help="select video input source for inference", default='rgb', choices=cam_options)

args = parser.parse_args()

cam_source = args.cam_input
nn_path = args.nn_model

nn_shape = 256
if '513' in nn_path:
    nn_shape = 513


def decode_deeplabv3p(output_tensor):
    class_colors = [[0, 0, 0], [0, 255, 0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(nn_shape, nn_shape)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors


def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame, 1, output_colors, 0.2, 0)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam=None
# Define a source - color camera
if cam_source == 'rgb':
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(nn_shape, nn_shape)
    cam.setInterleaved(False)
    cam.setFps(40)
    cam.preview.link(detection_nn.input)
elif cam_source == 'left':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
elif cam_source == 'right':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

else:
    # Define the input link
    xinFrame = pipeline.createXLinkIn()
    xinFrame.setStreamName("inFrame")
    xinFrame.out.link(detection_nn.input)


if cam_source in cam_options[1:]:
    manip = pipeline.createImageManip()
    manip.setResize(nn_shape, nn_shape)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)


# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)
detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)
detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
# device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
if cam_source not in cam_options:
    qIn = device.getInputQueue(name="inFrame")
    cap = cv2.VideoCapture(cam_source)

q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

start_time = time.time()
counter = 0
fps = 0
layer_info_printed = False
frame = None

while True:
    if cam_source not in cam_options:
        ret, frame = cap.read()
        if not ret:
            break
        # else:
        #     frames_counter +=1

        img = dai.ImgFrame()
        img.setData(to_planar(frame, (nn_shape, nn_shape)))
        img.setTimestamp(time.monotonic())
        img.setWidth(nn_shape)
        img.setHeight(nn_shape)
        qIn.send(img)
        frame = cv2.resize(frame, (nn_shape, nn_shape))

    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_nn = q_nn.get()
    if cam_source in cam_options:
        in_nn_input = q_nn_input.get()
        if in_nn_input is not None:
            # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
            shape = (3, in_nn_input.getHeight(), in_nn_input.getWidth())
            frame = in_nn_input.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame = np.ascontiguousarray(frame)
            print(frame.shape)

    if in_nn is not None:
        # print("NN received")
        layers = in_nn.getAllLayers()

        if not layer_info_printed:
            for layer_nr, layer in enumerate(layers):
                print(f"Layer {layer_nr}")
                print(f"Name: {layer.name}")
                print(f"Order: {layer.order}")
                print(f"dataType: {layer.dataType}")
                # reverse dimensions
                dims = layer.dims[::-1]
                print(f"dims: {dims}")
            layer_info_printed = True

        # get layer1 data
        layer1 = in_nn.getLayerInt32(layers[0].name)
        # reshape to numpy array
        dims = layer.dims[::-1]
        lay1 = np.asarray(layer1, dtype=np.int32).reshape(dims)

        output_colors = decode_deeplabv3p(lay1)

        if frame is not None:
            frame = show_deeplabv3p(output_colors, frame)
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
            cv2.imshow("nn_input", frame)

    counter+=1
    if (time.time() - start_time) > 1:
        fps = counter / (time.time() - start_time)
        counter = 0
        start_time = time.time()

    if cv2.waitKey(40) == ord('q'):
        break

if cam_source not in cam_options:
    cv2.destroyAllWindows()
    cap.release()
