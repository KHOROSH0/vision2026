import cv2
import numpy as np
from simple_canny import simple_canny

image = cv2.imread("../resources/kidsnoise.bmp")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered = cv2.medianBlur(gray, 3)


def update(_):
    low = cv2.getTrackbarPos("Low", window)
    high = cv2.getTrackbarPos("High", window)

    edges_approx = simple_canny(filtered, low, high)
    edges_fact = cv2.Canny(filtered, low, high, apertureSize=3)

    combined = np.hstack((edges_approx, edges_fact))
    cv2.imshow(window, combined)


window = "Simple | Opencv"
cv2.namedWindow(window)

cv2.createTrackbar("Low", window, 100, 255, update)

cv2.createTrackbar("High", window, 200, 255, update)

update(0)


cv2.waitKey(0)
