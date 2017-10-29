"""
数据收集、处理
1. 下载《爱情公寓》视频
2. 提取视频中的帧
3. 检测是否包含人脸
4. 包含则保存该帧图片
"""

import os
import cv2
import numpy as np
import imutils


def read_video(video_path, save_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError('File "%s" not found.' % video_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
    while cap.isOpened():
        index = cap.get(0)
        ret, frame = cap.read()
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y: y + h, x: x + w, :]
            file_name = '%s_%s.jpg' % (index, i)
            cv2.imwrite(os.path.join(save_path, file_name), face)
            print('File %s saved.' % file_name)
    cap.release()

PATH_VIDEO = u'E:/Youku Files/transcode/爱情公寓 第一季 01_超清.mp4'
PATH_SAVE = 'images'

if __name__ == '__main__':
    read_video(PATH_VIDEO, PATH_SAVE)

