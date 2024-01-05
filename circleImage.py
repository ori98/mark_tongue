import cv2


def circle_image():
    image = cv2.imread('androidFlask.jpg')
    image = cv2.circle(image, (50, 50), 30, (50, 200, 75), 5)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


circle_image()