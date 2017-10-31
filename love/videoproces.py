# coding=utf-8
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
import random
import data_handler
from v1.classifier import FaceClassifier

NAMES = data_handler.NAMES

def random_scale():
    offset_scale = random.random() - 0.5
    offset_scale += -0.5 if offset_scale < 0 else 0.5
    return offset_scale


def read_video(video_path, save_path, unknown=False, classify=False, classify_dir=None):
    if not os.path.exists(video_path):
        raise Exception('File "%s" not found.' % video_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if classify and classify_dir:
        classify_path = os.path.join(os.path.dirname(save_path), classify_dir)
    else:
        classify_path = None
    classifier = None
    cap = cv2.VideoCapture(video_path)
    cap.set(0, 2711617 * 0.2)
    face_cascade = cv2.CascadeClassifier('../config/haarcascade_frontalface_default.xml')
    while cap.isOpened():
        index = cap.get(0)
        ret, frame = cap.read()
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for i, (x, y, w, h) in enumerate(faces):
            if unknown:
                height = frame.shape[0]
                width = frame.shape[1]
                x += int(random_scale() * w)
                y += int(random_scale() * h)
                if x < 0 or y < 0 or x + w >= width or y + h >= height:
                    continue
            face = frame[y: y + h, x: x + w, :]
            file_name = '%s_%s.jpg' % (index, i)
            if classify:
                if classifier is None:
                    classifier = FaceClassifier()
                face = imutils.resize(face, 128, 128)
                if face.shape[0] == 128 and face.shape[1] == 128:
                    prediction = classifier.predict(np.array([face]))
                    index = np.argmax(prediction[0])
                    subdir = NAMES[index]
                    if classify_path:
                        save_path = os.path.join(classify_path, subdir)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, file_name), face)
            print('File %s saved.' % file_name)
    cap.release()


PATH_VIDEO = u'/Users/zijiao/Desktop/love1_3.mp4'
PATH_SAVE = 'images'

if __name__ == '__main__':
    read_video(PATH_VIDEO, PATH_SAVE, unknown=False, classify=True, classify_dir='roles')
