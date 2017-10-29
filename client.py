import numpy as np
import imutils
import time
from v1.classifier import Classifier
import love.data_handler as data_handler
import cv2

cap = cv2.VideoCapture('love1_1.mp4')
cap.set(0, int(8.85e5))

# cap = cv2.VideoCapture(0)
# cap.set(0, int(4.85e5))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = Classifier('v1/weights/weights.h5')

target_W = 128
target_H = 128
NAMES = ['Unknown', 'Guan', 'Lv', 'Zeng', 'Lin', 'Hu', 'Lu', 'Chen']

while (cap.isOpened()):
    ret, frame = cap.read()

    h_steps, v_steps = 1, 1
    if isinstance(frame, np.ndarray):
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        if len(faces):
            xs, ls = [], []
            for (x, y, w, h) in faces:
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = frame[y: y + h, x: x + w, :]
                face = imutils.resize(face, target_W, target_H)
                if face.shape[0] == target_H and face.shape[1] == target_W:
                    xs.append(face)
                    ls.append((x, y + 5 if y < 20 else y - 5))
            if len(xs) == 0:
                continue
            prediction = classifier.predict(np.array(xs))
            result = data_handler.parse_predict(prediction, NAMES)
            result = ['%s: %.2f' % (n, p) for n, p in result]
            for location, text in zip(ls, result):
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
