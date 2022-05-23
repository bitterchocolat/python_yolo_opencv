import cv2 as cv
import numpy as np
# import time
# import imutils


def yolo_vid_detect(net, size, input_vid, conf_threshold, nms_threshold) :
    writer = None
    W = input_vid.get(cv.CAP_PROP_FRAME_WIDTH)
    H = input_vid.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = input_vid.get(cv.CAP_PROP_FPS)

    while True :
        grabbed, frame = input_vid.read()

        if not grabbed :
            break

        vid_blob = cv.dnn.blobFromImage(frame, 1/255.0, (size, size), swapRB = True, crop = False)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        yolo_cv.setInput(vid_blob)
        cv_outs = yolo_cv.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for output in cv_outs :
            for detection in output :
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold :
                    # 객체의 너비, 높이, 중앙 좌표값
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")

                    # 객체 테두리의 좌상단 좌표값
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, int(width), int(height)])
        
        indexs = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        font = cv.FONT_HERSHEY_SIMPLEX
        if len(indexs) > 0 :
            for i in range(len(boxes)) :
                if i in indexs :
                    x, y, w, h = boxes[i]
                    class_name = str(classes[class_ids[i]])
                    label = f"{class_name}: {confidences[i] :.2f}"
                    color = colors[class_ids[i]]

                    # 객체 테두리, 텍스트, 출력
                    cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 35, y - 18), color, -1)
                    cv.putText(frame, label, (x, y - 5), font, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        
        if writer is None :
            fourcc = cv.VideoWriter_fourcc(*"DIVX")
            writer = cv.VideoWriter("results\processed.avi", fourcc, fps, (int(W), int(H)))

        writer.write(frame)

    writer.release()
    input_vid.release()



yolo_cv = cv.dnn.readNet("yolov3\yolov3.weights", "yolov3\yolov3.cfg")
size_list = [320, 416, 608]


classes = []
with open("yolov3\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

np.random.seed(30)
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# 비디오 테스트
vid = cv.VideoCapture("sources\dddd1.mp4")
input_sl = int(input("크기 입력(0~2) : "))
yolo_vid_detect(yolo_cv, size_list[input_sl], vid, 0.7, 0.4)

print("finished")
