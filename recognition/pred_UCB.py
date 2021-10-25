#!/usr/bin/env python3
import queue
import cv2
import depthai as dai
import numpy as np
from time import monotonic
import sys
import os
import blobconverter

sys.path.append(os.path.abspath('../'))
from utils import LDA


labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# video_path = '/home/israel/Downloads/UCB/clips/090/007/nm/nm-01.avi'
# videoPath = '../data/videos/001-bg-01-090.avi'
# video_path = '/home/israel/Downloads/OAKD_8S/clips/090/002/nm/nm-04.avi'

# video_path = '/home/israel/Downloads/CASIA/DatasetB-2/video/001-bg-01-090.avi'
video_path = '/home/israel/Downloads/CASIA/DatasetB-1/video/001-nm-01-090.avi'
# video_path = '/home/israel/Downloads/OAK2/j3/000-carlos1/nm.mp4'
input = 'vid'

if input == 'cam':
    nn1blob = '../models/mobilenet-ssd_openvino_2021.2_6shave.blob'
    # nn2blob = '/home/israel/Downloads/frozen_graph.blob'
elif input == 'vid':
    nn1blob = '../models/mobilenet-ssd_openvino_2021.2_8shave.blob'
    # nn2blob = '/home/israel/Downloads/frozen_graph.blob'

nn2blob = '../models/unet8shaves.blob'
# nn2blob = '../models/128_CASIA_Best.blob'
# nn2blob = '../models/128x128_acc_0.blob'
# nn2blob = '../models/128x128_acc_0.blob'
# nn2blob = '../models/128UCB.blob'
# nn2blob = '../models/UCB300/128x128unet_acc:0.9540_loss:0.0590_val-acc:0.9538_val-loss:0.0594_0.22M_01-08-21-DB_UCB300_E:10x1E-4:5x1E-5'
# nn2blob = '../models/64x64unet6shaves.blob'
# nn2blob = '/home/israel/Downloads/128x128_acc_1.blob'
color = (255, 0, 0)
width, height = 300, 300
size = 128
font = cv2.FONT_ITALIC
lastlayer = ''
silhouettes = []
nsil = 0


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def fine_mask(mask):
    ''' Takes a raw mask as input and returns the biggest contour mask '''
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    out = cv2.drawContours(out, [c], 0, 255, -1)
    return out


def colorCast(img):
    ''' Takes a raw image as input and reduces the saturation of GREEN channel'''
    rng = np.array([0, 32, 64, 96, 128, 160, 192, 224, 255])
    norm_rng = rng / (2000 /(1 + 5))
    exprng = np.exp(norm_rng)
    exprng = 255 * ((exprng - exprng.max()) / (exprng.max() - exprng.min()) + 1)
    # Create a LookUp Table
    fullRange = np.arange(0, 256)
    gLUT = np.interp(fullRange, rng, exprng)

    # Get the green channel and apply the mapping
    gChannel = img[:, :, 1]
    gChannel = cv2.LUT(gChannel, gLUT)
    img[:, :, 1] = gChannel
    return img


# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)


detection_nn = pipeline.createMobileNetDetectionNetwork()
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setBlobPath(nn1blob)
detection_nn.setNumInferenceThreads(2)
detection_nn.input.setBlocking(False)

# Define the second NN that will perform the segmentation
landmarks_nn = pipeline.createNeuralNetwork()
# landmarks_nn.setBlobPath('../models/64x64unet6shaves.blob')
# landmarks_nn.setBlobPath('../models/unet6shaves.blob')
landmarks_nn.setBlobPath(nn2blob)

# Create outputs
xin_rgb = pipeline.createXLinkIn()
xin_rgb.setStreamName("land_in")
xin_rgb.out.link(landmarks_nn.input)

# Create outputs
xout_frame = pipeline.createXLinkOut()
xout_frame.setStreamName("det_frame")

xout_det = pipeline.createXLinkOut()
xout_det.setStreamName("det_nn")
detection_nn.out.link(xout_det.input)

xout_land = pipeline.createXLinkOut()
xout_land.setStreamName("land_nn")
landmarks_nn.out.link(xout_land.input)

if input == 'cam':
    # Define a source - color camera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(width, height)
    cam_rgb.setInterleaved(False)
    cam_rgb.preview.link(detection_nn.input)
    cam_rgb.preview.link(xout_frame.input)


elif input == 'vid':
    xinFrame = pipeline.createXLinkIn()
    xinFrame.setStreamName("inFrame")
    xinFrame.out.link(detection_nn.input)
    xinFrame.out.link(xout_frame.input)


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    if input == 'vid':
        qIn = device.getInputQueue(name="inFrame")
        cap = cv2.VideoCapture(video_path)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        # delay = int(1000 / fps)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_frame = device.getOutputQueue(name="det_frame", maxSize=4, blocking=False)
    q_det = device.getOutputQueue(name="det_nn", maxSize=4, blocking=False)
    land_in = device.getInputQueue(name="land_in", maxSize=4, blocking=False)
    q_land = device.getOutputQueue(name="land_nn", maxSize=4, blocking=False)

    face_q = queue.Queue()
    size_q = queue.Queue()

    start_time = monotonic()
    nn1 = 0
    nn2 = 0
    fps1 = 0
    fps2 = 0
    cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
    while True:
        if (monotonic() - start_time) > 1:
            fps1 = nn1 // (monotonic() - start_time)
            fps2 = nn2 // (monotonic() - start_time)
            nn2 = 0
            nn1 = 0
            start_time = monotonic()

        if input == 'vid':
            read_correctly, frame = cap.read()
            # print(frame.shape)
            if not read_correctly:
                print('Breaking')
                break

            img = dai.ImgFrame()
            img.setData(to_planar(frame, (width, height)))
            img.setTimestamp(monotonic())
            img.setWidth(width)
            img.setHeight(height)
            qIn.send(img)

        while q_det.has():
            nn1 += 1
            in_frame = q_frame.get()
            shape = (3, in_frame.getHeight(), in_frame.getWidth())
            det_frame = in_frame.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame = np.ascontiguousarray(det_frame)
            # frame = colorCast(frame)
            copied = frame.copy()
            # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
            inDet = q_det.get()
            if inDet is not None and nn1%1==0:
                detections = inDet.detections
                if len(detections)>0:
                    for detection in detections:
                        if labelMap[detection.label] == 'person':
                            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                            x, y, w, h = bbox
                            offsets = [10, 10, 10, 10]
                            # offsets = [random.randint(0, w//4) for i in range(2)] + [random.randint(0, h//8) for i in range(2)]
                            x = x - offsets[0] if x - offsets[0] >=0 else 0
                            y = y - offsets[3] if y - offsets[3] >=0 else 0
                            w = w + offsets[1]+offsets[0] if w + offsets[1]+offsets[0] <=width else width
                            h = h + offsets[3]+offsets[1] if h + offsets[3]+offsets[1] <=height else height
                            bbox = [x, y, w, h]
                            # cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                            # cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                            cv2.putText(frame, f"NN1 Fps: {fps1}", (10, 10), font, 0.5, 255)
                            cv2.putText(frame, f"NN2 Fps: {fps2}", (10, 20), font, 0.5, 255)

                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                            face_frame = copied[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

                            nn_data = dai.NNData()
                            nn_data.setLayer("0", to_planar(face_frame, (size, size)))
                            land_in.send(nn_data)
                            face_q.put(face_frame)
                            size_q.put(bbox)
            cv2.imshow("rgb", frame)

            while q_land.has():
                nn2 += 1
                face_frame = face_q.get()
                org_size = size_q.get()
                x1, y1, x2, y2 = org_size
                w = x2 - x1
                h = y2 - y1

                inDet = q_land.get()
                if lastlayer == '':
                    lastlayer = inDet.getAllLayerNames()

                out = inDet.getLayerFp16(lastlayer[0])
                output = np.array(out).reshape(size, size)
                norm_image = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # norm_image = cv2.normalize(output,  None, 0, 255, cv2.NORM_MINMAX)
                # print(cv2.minMaxLoc(norm_image))

                # norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB)
                ret, mask = cv2.threshold(norm_image, 120, 255, cv2.THRESH_BINARY)
                mask = cv2.resize(mask, (w, h), cv2.INTER_CUBIC)
                mask = mask[5:h-5, 5:w-5]
                # print(org_size)
                # mask = fine_mask(mask)
                # h, w = mask.shape

                silhouettes.append(mask)
                # cv2.imwrite(f'{nsil}.png',mask)
                nsil +=1
                if nsil % 10 == 0:
                    GEI, _ = LDA.GEI_generator(silhouettes, debug=False)
                    cv2.imwrite(f'{nsil}.png',GEI)
                    classID = LDA.inference(GEI)
                    print(classID)
                # if 10<mask.mean() and mask.mean()<150:
                cv2.imshow('out', mask)

        if cv2.waitKey(1) == ord('q'):
            break
