import numpy as np
import cv2
import matplotlib.pyplot as plt
import random 


from task_1 import extension_image


def median_filter(img, ksize):
    shape1 = img.shape[0]
    shape2 = img.shape[1]
    # extense original image
    ext_img = extension_image(img, ksize//2)
    ext_shape1 = ext_img.shape[0]
    ext_shape2 = ext_img.shape[1]
    result = []
    result_iter = []
    # how many times we use a kernel
    N1 = ext_shape1 - ksize+ 1
    N2 = ext_shape2 - ksize + 1
    for i in range(N1):
        for j in range(N2):
            image_iter = ext_img[i:ksize+i, j:ksize+j]
            vector_iter = image_iter.flatten()
            sorted_image_iter = np.copy(sorted(vector_iter))
            pixel = sorted_image_iter[ksize**2//2]
            result_iter.append(pixel)
        result.append(np.array(result_iter))
        result_iter.clear()
    return np.array(result, dtype=np.uint8)






def main():

    # get an image
    image = cv2.imread("resources/kidsnoise.bmp")

    # convert to gray image
    img_u16 = image.astype(np.uint16)
    gray = (img_u16[:,:,0] + img_u16[:,:,1] + img_u16[:,:,2]) // 3
    gray = gray.astype(np.uint8)
    ksize = 5

    my_median_filter = median_filter(gray, ksize)
    cv2_median_filter = cv2.medianBlur(src=gray, ksize=ksize)

    plt.figure(1)
    plt.imshow(gray,cmap="gray", vmin=0, vmax=255)

    plt.figure(2)
    plt.imshow(my_median_filter,cmap="gray", vmin=0, vmax=255)

    plt.figure(3)
    plt.imshow(cv2_median_filter,cmap="gray", vmin=0, vmax=255)
    plt.show()    


if __name__ == "__main__":
    main()