import os.path

import cv2
import pathlib

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

# building a classifier
clf = cv2.CascadeClassifier(str(cascade_path))

# loading the image
img = cv2.imread("androidFlask.jpg")

# Using classifier to detect the face

# first converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# now detecting the face using classifier
faces = clf.detectMultiScale(
    gray,
    scaleFactor= 1.1,
    minNeighbors= 5,
    minSize= (30, 30),
    flags= cv2.CASCADE_SCALE_IMAGE
)

print("Number of faces detected", len(faces))

#looping over all the faces detected:
for(x,y, width, height) in faces:
    cv2.rectangle(img, (x, y), (x + width, y + width), (0, 255, 255), 2)

# while True: _, frame = camera.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = clf.detectMultiScale(
#         gray,
#         scaleFactor = 1.1,
#         minNeighbors = 5,
#         minSize = (30, 30),
#         flags = cv2.CASCADE_SCALE_IMAGE
#     )

cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()