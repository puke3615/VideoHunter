# coding=utf-8
import os
import cv2
import time
import utils
import numpy as np
import imutils
import random

from v1.classifier import FaceClassifier


def read_video(video_path, show=False, log=True, intercept=None, strides=1, title='Face Detection', seek=None):
    if strides <= 0:
        strides = 1
    utils.check_file(video_path)
    cap = cv2.VideoCapture(video_path)
    if seek is not None:
        cap.set(0, seek)
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame_index % strides == 0:
            if log:
                print('Progress %3.3f%%, Frame %s, %s' % (cap.get(2) * 100, frame_index, cap.get(0)))
            if intercept is not None:
                if isinstance(intercept, list):
                    for it in intercept:
                        frame = it(frame)
                else:
                    frame = intercept(frame)
        try:
            if show:
                cv2.imshow(title, frame)
                # cv2.waitKey(1000 / int(fps))  # 延迟
        except Exception as e:
            print(e)
            break
        frame_index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if show:
        cv2.destroyAllWindows()


def add_rect(im):
    return cv2.rectangle(im, (100, 100), (300, 300), (0, 0, 255), 2)


def save_image(im):
    faces = utils.detect_faces(im)
    save_path = PATH_SAVE
    for x, y, w, h in faces:
        face = im[y: y + h, x: x + w, :]
        file_name = '%s.jpg' % time.time()
        if CLASSIFY:
            face_data = imutils.resize(face, 128, 128)
            if face_data.shape[0] == 128 and face_data.shape[1] == 128:
                prediction = classifier.predict(np.array([face_data]))
                if np.max(prediction[0]) < FACE_MIN or np.max(prediction[0]) > FACE_MAX:
                    continue
                index = np.argmax(prediction[0])
                subdir = utils.NAMES_EN[index]
                save_path = os.path.join(CLASSIFY_PATH, subdir)
        utils.ensure_dir(save_path)
        final_path = os.path.join(save_path, file_name)
        cv2.imwrite(final_path, face)
        # print('File %s saved.' % file_name)
    return im


# 读取视频路径
# PATH_VIDEO = u'E:/Youku Files/transcode/爱情公寓 第一季 10_超清.mp4'
PATH_VIDEO = u'/Users/zijiao/Desktop/love1_3.mp4'
# 普通模式保存路径
PATH_SAVE = utils.root_path('data/love/images')
# 分类模式保存路径
CLASSIFY_PATH = utils.root_path('data/love/predict')
# seek位置
SEEK = 2.6005e6
# 识别阈值(0.0 ~ 1.0)
FACE_MIN = 0.0
FACE_MAX = 1.0
# 是否保存图片
SAVE_IMAGE = True
# 是否自动分类
CLASSIFY = True
# 每隔多少帧进行采样
STRIDES = 1
# 是否显示画面
SHOW_CONTENT = False
# 是否显示视频百分比
LOG = True

if __name__ == '__main__':
    classifier = FaceClassifier() if CLASSIFY else None
    read_video(PATH_VIDEO, strides=STRIDES, show=SHOW_CONTENT, log=LOG, intercept=save_image if SAVE_IMAGE else None, seek=SEEK)
