import cv2

image = cv2.imread("resources/kidsnoise.bmp")

cv2.imshow("img", image)
cv2.imshow("img2", image)
blurred = cv2.GaussianBlur(
        image,
        (5, 5),
        1.0
    )
cv2.imshow("blurred", blurred)
cv2.waitKey(0)