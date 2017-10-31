import numpy as np
import cv2
import os

NAMES = ['关谷神奇', '吕子乔', '曾小贤', '林宛瑜', '胡一菲', '陆展博', '陈美嘉']
FACE_PATH = 'config/haarcascade_frontalface_default.xml'


def parse_name(names=NAMES):
    index2name = {i: n for i, n in enumerate(names)}
    name2index = {v: k for k, v in index2name.items()}
    return names, index2name, name2index


def parse_predict(prediction, names=NAMES):
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
    while not root.endswith('VideoHunter'):
        root = os.path.dirname(root)
    for layer in path.split('/'):
        root = os.path.join(root, layer)
    return root


face_cascade = cv2.CascadeClassifier(root_path(FACE_PATH))


def detect_faces(im):
    return face_cascade.detectMultiScale(im, 1.1, 3)
