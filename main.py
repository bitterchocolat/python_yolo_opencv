import cv2 as cv
import numpy as np
import os


yolo_cv = cv.dnn.readNet("yolov3\yolov3.weights", "yolov3\yolov3.cfg")
layer_names = yolo_cv.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_cv.getUnconnectedOutLayers()]


classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
