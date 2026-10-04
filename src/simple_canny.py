import cv2
import numpy as np


def non_maximum_suppression(magnitude, angle):
    h, w = magnitude.shape
    result = np.zeros_like(magnitude, dtype=np.float32)

    # Convert radians to degrees and map to [0, 180)
    angle = np.rad2deg(angle)
    angle[angle < 0] += 180

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            m = magnitude[y, x]
            a = angle[y, x]

            # Gradient direction ≈ 0 degrees
            if a < 22.5 or a >= 157.5:
                m1 = magnitude[y, x - 1]
                m2 = magnitude[y, x + 1]
                # Keep only local maxima
                if m > m1 and m >= m2:
                    result[y, x] = m

            # Gradient direction ≈ 45 degrees
            elif a < 67.5:
                m1 = magnitude[y + 1, x + 1]
                m2 = magnitude[y - 1, x - 1]
                # Keep only local maxima
                if m > m1 and m > m2:
                    result[y, x] = m

            # Gradient direction ≈ 90 degrees
            elif a < 112.5:
                m1 = magnitude[y - 1, x]
                m2 = magnitude[y + 1, x]
                # Keep only local maxima
                if m > m1 and m >= m2:
                    result[y, x] = m

            # Gradient direction ≈ 135 degrees
            else:
                m1 = magnitude[y + 1, x - 1]
                m2 = magnitude[y - 1, x + 1]
                # Keep only local maxima
                if m > m1 and m > m2:
                    result[y, x] = m

    return result


def hysteresis(image, low_threshold, high_threshold):
    h, w = image.shape

    result = np.zeros((h, w), dtype=np.uint8)

    strong = image >= high_threshold

    weak = (image >= low_threshold) & (image < high_threshold)

    # Strong pixels are immediately accepted
    result[strong] = 255

    # Start search from all strong pixels
    stack = list(zip(*np.nonzero(strong)))

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while stack:
        y, x = stack.pop()

        for dy, dx in neighbors:
            ny = y + dy
            nx = x + dx

            if not (0 <= ny < h and 0 <= nx < w):
                continue

            # A weak pixel connected to an accepted edge
            # becomes an accepted edge itself.
            if weak[ny, nx] and result[ny, nx] == 0:
                result[ny, nx] = 255
                stack.append((ny, nx))

    return result


def simple_canny(image, low_threshold, high_threshold, L2gradient=False):

    # 1. Gaussian blur
    # OpenCV implementation actually does not include blurring

    # -------------------------------------------------
    # 2. Image gradients
    # -------------------------------------------------

    gx = cv2.Sobel(
        image, cv2.CV_16S, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE
    ).astype(np.float32)

    gy = cv2.Sobel(
        image, cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE
    ).astype(np.float32)

    # Gradient magnitude
    if L2gradient:
        magnitude = np.sqrt(gx**2 + gy**2)
    else:
        magnitude = np.abs(gx) + np.abs(gy)

    # Gradient direction
    angle = np.arctan2(gy, gx)

    # -------------------------------------------------
    # 3. Non-maximum suppression
    # -------------------------------------------------

    nms = non_maximum_suppression(magnitude, angle)

    # -------------------------------------------------
    # 4 + 5. Double threshold + hysteresis
    # -------------------------------------------------

    edges = hysteresis(nms, low_threshold, high_threshold)

    return edges