import cv2 as cv
import numpy as np
# import time
# import imutils


def yolo_img_detect(net, size, input_img, conf_threshold, nms_threshold) :
    
    draw_img = input_img.copy()
    height, width = input_img.shape[:2]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # 네트워크 이미지 전처리, 입력, 출력
    blob = cv.dnn.blobFromImage(draw_img, 1/255.0, (size, size), (0, 0, 0), True, False)
    net.setInput(blob)
    cv_outs = net.forward(output_layers)


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
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 객체 테두리의 좌상단 좌표값
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Non Maximum Suppression
    indexs = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)


    font = cv.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)) :
        if i in indexs :
            x, y, w, h = boxes[i]
            class_name = str(classes[class_ids[i]])
            label = f"{class_name}: {confidences[i] :.2f}"
            color = colors[i]

            # 객체 테두리, 텍스트, 출력
            cv.rectangle(draw_img, (x, y), (x + w, y + h), color, 2)
            cv.rectangle(draw_img, (x - 1, y), (x + len(class_name) * 13 + 35, y - 18), color, -1)
            cv.putText(draw_img, label, (x, y - 5), font, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    return draw_img




yolo_cv = cv.dnn.readNet("yolov3\yolov3.weights", "yolov3\yolov3.cfg")
size_list = [320, 416, 608]


classes = []
with open("yolov3\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))


# 이미지 테스트
img = cv.imread("sources\img6.jpg")
img = cv.resize(img, None, fx = 0.5, fy = 0.5)
input_sl = int(input("크기 입력(0~2) : "))
result_img = yolo_img_detect(yolo_cv, size_list[input_sl], img, 0.7, 0.4)
cv.imwrite("results\processed_img.jpg", result_img)
cv.imshow("result", result_img)
cv.waitKey(0)
cv.destroyAllWindows()

print("finished")
