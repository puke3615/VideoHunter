# coding=utf-8
import os
import cv2
import numpy as np
import imutils
import random

# PATH_VIDEO = u'/Users/zijiao/Desktop/love1_3.mp4'
PATH_VIDEO = u'/Users/zijiao/Desktop/love1_3.mp4'
PATH_SAVE = 'images'

if __name__ == '__main__':
    cap = cv2.VideoCapture(PATH_VIDEO)
    fps = 24  # 视频帧率
    w = int(cap.get(3))
    h = int(cap.get(4))

    # 获得码率及尺寸
    # fps = cap.get(cv2.CV_CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH)),
    #         int(cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)))

    # 指定写视频的格式, I420-avi, MJPG-mp4
    videoWriter = cv2.VideoWriter('./3.mp4', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

    face_cascade = cv2.CascadeClassifier('../config/haarcascade_frontalface_default.xml')
    c = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # frame = imutils.resize(frame, 300)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y: y + h, x: x + w, :]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (00, 00, 255), 2)
        try:
            # cv2.imshow('Face Detection', frame)

            # cv2.waitKey(1000 / int(fps))  # 延迟
            videoWriter.write(frame)  # 写视频帧
            print 'Progress %3.3f%%, Frame %s' % (cap.get(2) * 100, c)
            c += 1
        except Exception as e:
            print(e)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoWriter.release()
    cap.release()
    cv2.destroyAllWindows()
