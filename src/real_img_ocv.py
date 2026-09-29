import cv2
import numpy as np

def filter2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ksize = kernel.shape[0]
    pad = ksize //2

    kernel = kernel.astype(np.float32)

    # Replicate border pixels
    padded = np.pad(
        image,
        ((pad, pad), (pad, pad)),
        mode="edge"
    ).astype(np.float32)

    output = np.zeros(image.shape, dtype=np.float32)
    height, width = image.shape
    # Apply kernel
    for y in range(height):
        for x in range(width):
            region = padded[
                y:y + ksize,
                x:x + ksize
            ]
            output[y, x] = np.sum(region * kernel)

    return output

# Sobel kernel
edge_kern_v = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# img_sim = np.array([
#     [3,0,1],
#     [3,0,1],
#     [3,0,1]
# ], dtype=np.uint8)

# print(f'sim: {np.sum(img_sim * edge_kern_v)}')

edge_kern_h = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

image = cv2.imread("../resources/kidsnoise.bmp")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered = cv2.medianBlur(gray, 3)
# filtered = cv2.GaussianBlur(gray, (5,5),1.0)
filtered_f = filtered.astype(np.float32)
# edges_v = cv2.filter2D(filtered_f, -1, kernel=edge_kern_v).astype(np.float32)
# edges_h = cv2.filter2D(filtered_f, -1, kernel=edge_kern_h).astype(np.float32)
# edges_v = filter2d(filtered, kernel=edge_kern_v).astype(np.float32)
# edges_h = filter2d(filtered, kernel=edge_kern_h).astype(np.float32)

# Most specific Sobel opencv
edges_v = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3) # dx=1, dy=0
edges_h = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3) # dx=0, dy=1

diff_gain = 1.0
edges = np.sqrt(edges_v**2 + edges_h**2)
edges = np.clip(diff_gain*edges, 0, 255).astype(np.uint8)

ret,thresh = cv2.threshold(edges,127,255,cv2.THRESH_BINARY)

# Display part
# cv2.imshow("gray", gray)
cv2.imshow("filtered", filtered)
# cv2.imshow("edges_v", edges_v)
# cv2.imshow("edges_h", edges_h)
cv2.imshow("edges", edges)
cv2.imshow("thresholded", thresh)
cv2.waitKey(0)