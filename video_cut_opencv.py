import cv2 as cv


iCurrent_frame = 0
start_frame = int(input("숫자 입력 : "))
end_frame = int(input("숫자 입력 : "))


vid = cv.VideoCapture("sources\_pre.mp4")
W = vid.get(cv.CAP_PROP_FRAME_WIDTH)
H = vid.get(cv.CAP_PROP_FRAME_HEIGHT)
fps = vid.get(cv.CAP_PROP_FPS)

writer = None

vid.set(cv.CAP_PROP_POS_FRAMES, start_frame)

while True :

    if iCurrent_frame > (end_frame - start_frame) :
        break
    iCurrent_frame = iCurrent_frame + 1
    
    grabbed, frame = vid.read()

    if not grabbed :
        break


    if writer is None :
        fourcc = cv.VideoWriter_fourcc(*"DIVX")
        writer = cv.VideoWriter("results\_processed.avi", fourcc, fps, (int(W), int(H)))

    writer.write(frame)

writer.release()
vid.release()

print("finished")
