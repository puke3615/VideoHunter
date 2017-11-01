# coding=utf-8
import cv2
import utils
import imutils
import numpy as np

from v1.classifier import FaceClassifier

# VIDEO_PATH = u'E:/Youku Files/transcode/爱情公寓 第一季 08_超清.mp4'
VIDEO_PATH = u'/Users/zijiao/Desktop/love1_3.mp4'
PROB_THRESHOLD = 0.5
SEEK = 5.85e5

if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(0, int(15.85e5))

    classifier = FaceClassifier()

    target_W = utils.IM_WIDTH
    target_H = utils.IM_HEIGHT

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not isinstance(frame, np.ndarray):
            continue
        faces = utils.detect_faces(frame)
        if len(faces):
            xs, ls = [], []
            for (x, y, w, h) in faces:
                face = frame[y: y + h, x: x + w, :]
                face = imutils.resize(face, target_W, target_H)
                if face.shape[0] == target_H and face.shape[1] == target_W:
                    xs.append(face)
                    ls.append((x, y + 5 if y < 20 else y - 5))
            if xs:
                prediction = classifier.predict(np.array(xs))
                result = utils.parse_predict(prediction, utils.NAMES_EN)
                result = filter(lambda r: r[1] > PROB_THRESHOLD, result)
                result = ['%s: %.2f' % (n, p) for n, p in result]
                for location, text, (x, y, w, h) in zip(ls, result, faces):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, text, location,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        try:
            cv2.imshow('Face Detection', frame)
            # time.sleep(0.025)
        except Exception as e:
            print(e)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
