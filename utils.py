# coding=utf-8
import numpy as np
import cv2
import os

NAMES = [u'关谷神奇', u'吕子乔', u'曾小贤', u'林宛瑜', u'胡一菲', u'陆展博', u'陈美嘉']
NAMES_EN = ['Guan', 'Lv', 'Zeng', 'Lin', 'Hu', 'Lu', 'Chen']
FACE_PATH = 'config/haarcascade_frontalface_alt.xml'
PROJECT_NAME = 'VideoHunter'
IM_WIDTH = 128
IM_HEIGHT = 128


def parse_name(names=NAMES):
    index2name = {i: n for i, n in enumerate(names)}
    name2index = {v: k for k, v in index2name.items()}
    return names, index2name, name2index


def parse_predict(prediction, names=NAMES, unique=False):
    if unique:
        prediction = np.copy(prediction)
        r = np.zeros((len(prediction), 2))
        while np.max(prediction) > 0:
            argmax = np.argmax(prediction)
            v_max = np.max(prediction)
            c_max = argmax % len(names)
            r_max = argmax / len(names)
            r[r_max, :] = [c_max, v_max]
            prediction[:, c_max] = 0
            prediction[r_max, :] = 0
        return [(names[int(class_index)], prob) for class_index, prob in r]
    else:
        result_index = np.argmax(prediction, 1).tolist()
        result_prob = np.max(prediction, 1).tolist()
        return [(names[index], prob) for index, prob in zip(result_index, result_prob)]


def check_file(file_path):
    if not os.path.exists(file_path):
        raise Exception('File "%s" not found.' % file_path)


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def root_path(path):
    root = os.path.abspath('.')
    while not root.endswith(PROJECT_NAME):
        root = os.path.dirname(root)
    for layer in path.split('/'):
        root = os.path.join(root, layer)
    return root


def calculate_file_num(dir):
    if not os.path.exists(dir):
        return 0
    if os.path.isfile(dir):
        return 1
    count = 0
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        count += calculate_file_num(sub_path)
    return count


face_cascade = cv2.CascadeClassifier(root_path(FACE_PATH))


def detect_faces(im):
    return face_cascade.detectMultiScale(im, 1.2, 3)
