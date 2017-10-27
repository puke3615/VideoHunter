import numpy as np
import imutils
import cv2

cap = cv2.VideoCapture('1.mp4')

target_W = 192 * 2
target_H = 108 * 2

while (cap.isOpened()):
    ret, frame = cap.read()

    h_steps, v_steps = 1, 1
    if isinstance(frame, np.ndarray):
        frame = imutils.resize(frame, width=target_W)

    try:
        cv2.imshow('frame', frame)
    except Exception as e:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
