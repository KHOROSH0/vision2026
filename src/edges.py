import cv2
from simple_canny import simple_canny

image = cv2.imread("../resources/kidsnoise.bmp")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered = cv2.medianBlur(gray, 3)

edges_approx = simple_canny(filtered, 100, 200)
edges_fact = cv2.Canny(filtered, 100, 200, apertureSize=3)

cv2.imshow("edges_approx", edges_approx)
cv2.imshow("edges_fact", edges_fact)
cv2.waitKey(0)