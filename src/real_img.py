import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("resources/kidsnoise.bmp")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
## the same effect:
# image = np.flip(image, 2)

## Simplest RGB to grayscale conversion
# img_u16 = image.astype(np.uint16)
# gray = (img_u16[:,:,0] + img_u16[:,:,1] + img_u16[:,:,2]) // 3
# gray = gray.astype(np.uint8)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.blur(gray, (3,3))
gauss_blur = cv2.GaussianBlur(gray, (7,7), 0.5)
medfilt = cv2.medianBlur(gray, 3)

print("Shape:", image.shape)
print("Data type:", image.dtype)

print("Gray Shape:", gray.shape)
print("Gray Data type:", gray.dtype)

## Image outputs

# plt.figure("orig")
# plt.imshow(image)

plt.figure("grayscale")
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.figure("blurred")
plt.imshow(blurred, cmap="gray", vmin=0, vmax=255)
plt.show()
# plt.figure("gauss blurred")
# plt.imshow(gauss_blur, cmap="gray", vmin=0, vmax=255)

# plt.figure("median filter")
# plt.imshow(medfilt, cmap="gray", vmin=0, vmax=255)

# plt.show()


## Playing with gaussian

# Define your kernel size (must be an odd and positive integer) and sigma
ksize = 5
sigma = 0.3

# 1. Generate the 1D Gaussian kernel (returns a ksize x 1 column vector)
kernel_1d = cv2.getGaussianKernel(ksize, sigma)

# 2. Compute the outer product to get the 2D Gaussian kernel
# Using the matrix multiplication operator '@'
kernel_2d = kernel_1d @ kernel_1d.T

print("2D Gaussian Kernel:")
# print(f'{kernel_2d:.4f}')

formatter = np.vectorize(lambda x: f"{x:.6f}")
string_arr = formatter(kernel_2d)

print(string_arr)