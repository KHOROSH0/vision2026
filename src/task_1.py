import numpy as np
import cv2
import matplotlib.pyplot as plt
import random 


# explode our matrix
def extension_image(img, size):
    shape1 = img.shape[0]
    shape2 = img.shape[1]
    result = np.zeros((shape1 + 2*(size), shape2 + 2*(size)))
    for i in range(shape1 + size):
        for j in range(shape2 + size):
            # condition for extension matrix
            if ((i < (size)) or (i > (shape1 + size))):
                continue
            if ((j < (size)) or (j > (shape2 + size))):
                continue
            # put value of original image in result (another step)
            else:
                result[i][j] = img[i-size][j - size]
    return result
                
        
# filter
def filter_2d(img, kernel: np.ndarray):
    
    shape1 = img.shape[0]
    shape2 = img.shape[1]
    kernel_shape = kernel.shape[0]
    # extense original image
    ext_img = extension_image(img, kernel.shape[0]//2)
    ext_shape1 = ext_img.shape[0]
    ext_shape2 = ext_img.shape[1]
    result = []
    result_iter = []
    # how many times we use a kernel
    N1 = ext_shape1 - kernel_shape + 1
    N2 = ext_shape2 - kernel_shape + 1
    for i in range(N1):
        for j in range(N2):
            image_iter = ext_img[i:kernel_shape+i, j:kernel_shape+j]
            pixel = np.sum(image_iter*kernel)
            result_iter.append(abs(pixel))
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



    # a random kernel
    test_kernel = [[random.random() for _ in range(8)] for _ in range(8)]
    for row in test_kernel:
        print(row)
    test_kernel = np.array(test_kernel/np.sum(test_kernel))

    sharpe_kernel = np.array([[-1,0,1], [-2, 0, 2],[-1,0,1]])
    sharpe_kernel_2 = np.array([[-1,-2,-1], [0,0,0],[1,2,1]])
    # draw graphs
    random_filter = filter_2d(gray, test_kernel)
    random_filter_2 = filter_2d(random_filter, sharpe_kernel)
    random_filter_3 = filter_2d(random_filter, sharpe_kernel_2)
    cv_filter = cv2.filter2D(src = gray, ddepth=-1, kernel=sharpe_kernel)
    plt.figure(1)
    plt.imshow(gray,cmap="gray", vmin=0, vmax=255)

    plt.figure(2)
    plt.imshow(random_filter_2,cmap="gray", vmin=0, vmax=255)

    plt.figure(3)
    plt.imshow(cv_filter,cmap="gray", vmin=0, vmax=255)
    plt.show()



if __name__ == "__main__":
    main()