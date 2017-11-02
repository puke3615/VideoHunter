# coding=utf-8
import os
import cv2
import utils
import imutils
import numpy as np
from PIL import Image
from v1.classifier import FaceClassifier

IMAGE_PATH = utils.root_path('doc/images/test6.jpg')
PREDICTION_POSTFIX = '_prediction'
SAVE_PREDICTION = True
PROB_THRESHOLD = 0.5

if __name__ == '__main__':
    classifier = FaceClassifier()
    target_W = utils.IM_WIDTH
    target_H = utils.IM_HEIGHT
    im = Image.open(IMAGE_PATH)
    im = np.array(im)
    faces = utils.detect_faces(im)

    if len(faces):
        xs, ls = [], []
        for (x, y, w, h) in faces:
            face = im[y: y + h, x: x + w, :]
            face = imutils.resize(face, target_W, target_H)
            if face.shape[0] == target_H and face.shape[1] == target_W:
                xs.append(face)
                ls.append((x, y + 5 if y < 20 else y - 5))
        prediction = classifier.predict(np.array(xs))
        result = utils.parse_predict(prediction, utils.NAMES_EN, unique=False)
        result = filter(lambda r: r[1] > PROB_THRESHOLD, result)
        result = ['%s: %.2f' % (n, p) for n, p in result]
        if result:
            for location, text, (x, y, w, h) in zip(ls, result, faces):
                img = cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(im, text, location,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    im = Image.fromarray(im)
    im.show()

    if SAVE_PREDICTION:
        prediction_dir = os.path.dirname(IMAGE_PATH)
        basenames = os.path.basename(IMAGE_PATH).split('.')
        prediction_filename = '%s%s.%s' % (basenames[0], PREDICTION_POSTFIX, basenames[1])
        im.save(os.path.join(prediction_dir, prediction_filename))
