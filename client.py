import numpy as np
import imutils
import time
import cv2

cap = cv2.VideoCapture('wolf.mp4')
cap.set(0, int(4.85e5))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

target_W = 1328

while (cap.isOpened()):
    ret, frame = cap.read()

    h_steps, v_steps = 1, 1
    if isinstance(frame, np.ndarray):
        frame = imutils.resize(frame, width=target_W)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.rectangle(frame, (100, 100), (300, 300), 0xff0000, 2)

        cv2.putText(frame, str(cap.get(0)), (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0xff0000, 2)

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = frame[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    try:
        cv2.imshow('frame', frame)
        time.sleep(0.025)
    except Exception as e:
        print(e)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
