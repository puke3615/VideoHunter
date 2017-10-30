import cv2

face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')
im = cv2.imread('test.jpg')


faces = face_cascade.detectMultiScale(im, 1.3, 5)
for x, y, w, h in faces:
    cv2.rectangle(im, (x, y), (x + w, y + h), 128, 2)

cv2.imshow('Face Detection', im)
cv2.waitKey(0)
cv2.destroyAllWindows()