import cv2
import numpy as np
from simple_canny import simple_canny
from connected_components import connected_components, component_area

def draw_components_random_colors(image, components):
    result = image.copy()

    for x, y in components[0]:
        result[y, x] = [0,0,255]

    for x, y in components[1]:
        result[y, x] = [0,255,0]

    return result

# image = cv2.imread("../resources/kidsnoise.bmp")
image = cv2.imread("../resources/figures.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# filtered = cv2.medianBlur(gray, 3)

low = 200
high = 250

edges_approx = simple_canny(gray, low, high)
components, labels = connected_components(edges_approx)

contours_img = draw_components_random_colors(image, components)

for component in components:
    print("---COMP---")
    area_fact = component_area(component, edges_approx.shape)
    print(f's_f = {area_fact}')
    perimeter = len(component)
    a = (perimeter + 4)/4
    area_theor = a**2
    print(f's_t = {area_theor}')
    print(f's_f/s_t = {area_fact/area_theor}')

# contours, hierarchy = cv2.findContours(
#         edges_approx,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_NONE
#     )
# print(len(contours))

# area = cv2.contourArea(contours[1])
# print(area)
# area = cv2.contourArea(contours[0])
# print(area)
# perimeter = cv2.arcLength(contours[0], closed=True)
# print((perimeter/4)**2)

cv2.imshow("contours", contours_img)
cv2.waitKey(0)
