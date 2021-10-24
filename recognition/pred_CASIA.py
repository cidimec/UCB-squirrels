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
# videoPath = '/home/israel/Downloads/CASIA/DatasetB-1/video/010-nm-05-090.avi'
videoPath = '../CASIA_utils/nm_1-30_cond.avi'
save_vid = False
show_output = True
# videoPath = '../data/videos/010-nm-01-090.avi'
# videoPath = '/home/israel/Downloads/OAK/j2/05.08.2021_12.42.19_id:carlos1_walk:nm/color_video.mp4'
# id_label = int(videoPath.split('/')[-1].split('-')[0])
id_label = 1
bsub = bsub()
predictions = []
# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

# Define the detection model and their properties
nn_shape = 300
nn = pipeline.createMobileNetDetectionNetwork()
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Define a source - color camera
cam_options = ['rgb', 'left', 'right']
cam_source = 'vid'
if cam_source == 'rgb':
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(nn_shape, nn_shape)
    cam.setInterleaved(False)
    cam.setFps(60)
    cam.preview.link(nn.input)
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
    xinFrame.out.link(nn.input)

# xinFrame = pipeline.createXLinkIn()
# xinFrame.setStreamName("inFrame")

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("input")
xout_rgb.input.setBlocking(False)
# Linking
if True:
    nn.passthrough.link(xout_rgb.input)
else:
    cam.preview.link(xout_rgb.input)

nnOut = pipeline.createXLinkOut()
nnOut.setStreamName("nn")
nnOut.input.setBlocking(False)
nn.out.link(nnOut.input)

# cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
# cv2.moveWindow('rgb', 0, 0)


# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)

if cam_source not in cam_options:
    qIn = device.getInputQueue(name="inFrame")
    cap = cv2.VideoCapture(videoPath)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # delay = int(1000 / fps)

# Input queue will be used to send video frames to the device.
q_nn_input = device.getOutputQueue(name="input", maxSize=4, blocking=False)
qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

startTime = monotonic()
cam_fps = monotonic()
cam_fps_mean = 60
nn_counter = 0
frames_counter = 0
frame = None
detections = []

font = cv2.FONT_ITALIC
colors = {'white':(255, 255, 255), 'red':(0,0,255), 'green':(0, 255, 0),'blue':(255, 0, 0)}


if save_vid:
    clip_name = '001-nm.mp4'
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fwidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(clip_name, codec, fps, (fwidth, fheight))


def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def displayFrame(name, frame):
    # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    current = frame.copy()
    bbox = None
    roi = np.zeros((64, 64, 3))
    for detection in detections:
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        if show_output:
            label = 'ID: {}, PRED: {}'.format(id_label, int(bsub.classID))
            color = colors['green'] if id_label == int(bsub.classID) else colors['red']
            cv2.putText(frame, label, (bbox[0], bbox[1] - 7), font, 0.5, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    # cv2.putText(frame, "Cam fps: {:.2f}".format(frames_counter / (monotonic() - startTime)), (2, frame.shape[0] - 15), font, 0.4, colors['white'])
    if show_output:
        cv2.putText(frame, "Fps: {:.2f}".format(nn_counter / (monotonic() - startTime)), (2, frame.shape[0] - 4), font, 0.4, colors['white'])
    # else:
        # print(f'Processing at : {nn_counter // (monotonic() - startTime)} fps')
    if bsub.bg is None:
        bsub.setBackound(current)
    elif bbox is not None:
        roi = bsub.substract(current, bbox, mode='deep')
        roi = cv2.resize(roi, (64, 64), cv2.INTER_AREA)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        frame[-64:,-64:] = roi.copy()
    # Show the frame
    if show_output: cv2.imshow(name, frame)
    if save_vid: out.write(frame)
    # if roi is not None:
        # cv2.imshow('roi', roi)

inRgb = None
inDet = None
while True:
    # Get the current image whether from video file or camera
    if cam_source not in cam_options:
        read_correctly, frame = cap.read()

        if not read_correctly:
            break
        if frame.mean() > 220:
            if len(bsub.pred_list)>0:
                predictions.append(bsub.pred_list)
            print('New video')
            bsub.clear()
        img = dai.ImgFrame()
        img.setData(to_planar(frame, (300, 300)))
        img.setTimestamp(monotonic())
        img.setWidth(300)
        img.setHeight(300)
        qIn.send(img)
    else:
        inRgb = q_nn_input.get()
        if inRgb is not None:
            frame = inRgb.getCvFrame()

    # Sppeds up the inference, but it loses accuraccy in the localization
    if frames_counter % 1 == 0:
        inDet = qDet.get()

    if inDet is not None:
        detections = inDet.detections
        nn_counter +=1

    if frame is not None:
        frames_counter +=1

        displayFrame("rgb", frame)

    if cv2.waitKey(1) == ord('q'):
        break

if cam_source not in cam_options:
    cv2.destroyAllWindows()
    cap.release()
    if save_vid: out.release()

print(predictions)
