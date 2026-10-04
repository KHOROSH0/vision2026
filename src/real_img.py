import cv2
import matplotlib.pyplot as plt
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

    # Saturate to uint8 range
    return np.clip(output, 0, 255).astype(np.uint8)

def median_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    pad = ksize // 2

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad)),
        mode="edge"
    )

    output = np.zeros_like(image)

    height, width = image.shape

    for y in range(height):
        for x in range(width):
            region = padded[
                y:y + ksize,
                x:x + ksize
            ]
            output[y, x] = np.median(region)

    return output

image = cv2.imread("../resources/kidsnoise.bmp")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
## the same effect:
# image = np.flip(image, 2)

## Simplest RGB to grayscale conversion
# img_u16 = image.astype(np.uint16)
# gray = (img_u16[:,:,0] + img_u16[:,:,1] + img_u16[:,:,2]) // 3
# gray = gray.astype(np.uint8)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur_k = 7
blurred = cv2.blur(gray, (blur_k, blur_k))
gauss_blur = cv2.GaussianBlur(gray, (7,7), 0.5)
medfilt = cv2.medianBlur(gray, 3)

print("Shape:", image.shape)
print("Data type:", image.dtype)

print("Gray Shape:", gray.shape)
print("Gray Data type:", gray.dtype)

blurred_func = filter2d(gray, np.ones((blur_k, blur_k))/blur_k**2)

# kernel examples:
avg_blur = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

eye_kern = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

cross_blur = np.array([
    [0,   1/5, 0],
    [1/5, 1/5, 1/5],
    [0,   1/5, 0]
])

# 5*(eye - cross)
sharp_kern = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
])

## Image outputs

# plt.figure("orig")
# plt.imshow(image)

# plt.figure("grayscale")
# plt.imshow(gray, cmap="gray", vmin=0, vmax=255)

plt.figure("blurred")
plt.imshow(blurred, cmap="gray", vmin=0, vmax=255)

plt.figure("blurred_func")
plt.imshow(blurred_func, cmap="gray", vmin=0, vmax=255)

sharpened_img = filter2d(blurred, sharp_kern)
plt.figure("sharpened")
plt.imshow(sharpened_img, cmap="gray", vmin=0, vmax=255)

# plt.figure("gauss blurred")
# plt.imshow(gauss_blur, cmap="gray", vmin=0, vmax=255)

# plt.figure("median filter")
# plt.imshow(medfilt, cmap="gray", vmin=0, vmax=255)

plt.show()


# ## Playing with gaussian

# # Define your kernel size (must be an odd and positive integer) and sigma
# ksize = 5
# sigma = 0.3

# # 1. Generate the 1D Gaussian kernel (returns a ksize x 1 column vector)
# kernel_1d = cv2.getGaussianKernel(ksize, sigma)

# # 2. Compute the outer product to get the 2D Gaussian kernel
# # Using the matrix multiplication operator '@'
# kernel_2d = kernel_1d @ kernel_1d.T

# print("2D Gaussian Kernel:")
# # print(f'{kernel_2d:.4f}')

# formatter = np.vectorize(lambda x: f"{x:.6f}")
# string_arr = formatter(kernel_2d)

# print(string_arr)