from collections import deque
import cv2
from task_1_edg import get_contours
import numpy as np
import matplotlib.pyplot as plt
def calc_neighbours(edges):
    matrix = []
    row = []
    element = []
    coordinate_of_element = []
    n = len(edges)
    m = len(edges[0])
    for i in range(n):
        for j in range(m):
            #left
            if ((i - 1 >=0) and edges[i-1][j] < 250): # into map
                coordinate_of_element.append(i-1)
                coordinate_of_element.append(j)
                element.append(coordinate_of_element.copy())
                coordinate_of_element.clear()

            #down
            if ((j-1 >=0) and (edges[i][j-1] < 250)): # into map
                coordinate_of_element.append(i)
                coordinate_of_element.append(j-1)
                element.append(coordinate_of_element.copy())
                coordinate_of_element.clear()
            #right
            if ((i+1 <n) and (edges[i+1][j] < 250)): # into map
                coordinate_of_element.append(i+1)
                coordinate_of_element.append(j)
                element.append(coordinate_of_element.copy())
                coordinate_of_element.clear()
            #up
            if ((j+1 <m) and (edges[i][j+1] < 250)): # into map
                coordinate_of_element.append(i)
                coordinate_of_element.append(j+1)
                element.append(coordinate_of_element.copy())
                coordinate_of_element.clear()
                # add element                                            
            row.append(element.copy())
            element.clear()
        # add row of elements
        matrix.append(row.copy())
        row.clear()
    return matrix
def bfs_square(neighbours, coord_start):
    q = deque()
    graph = []
    q.append(coord_start)
    n = len(neighbours)
    m = len(neighbours[0])
    dist = [[m*n for _ in range(m)] for _ in range(n)]
    i,j = coord_start
    dist[i][j] = 0
    square = 1
    while (len(q) != 0):
        v = q.popleft()
        i,j = v
        for k in neighbours[i][j]:
            if (dist[k[0]][k[1]] > dist[i][j] + 1):
                dist[k[0]][k[1]] = dist[i][j]  + 1
                square+=1
                q.append(k)

    return square

def area(contour):
    n = max(contour, key = lambda x: x[0])[0] + 10
    m = max(contour, key = lambda x: x[1])[1]  + 10
    min_n = min(contour, key = lambda x: x[0])[0]
    min_m = min(contour, key = lambda x: x[1])[1]
    start_matrix = np.array([[0 for _ in range(m)] for _ in range(n)])
    # find neighbours using lets
    coord_start = [int((n + min_n)/2), int((m + min_m)/2)]
    start_matrix[coord_start[0]][coord_start[1]] = 255
    for i in range(n):
        for j in range(m):
            coord = [i,j]
            if ((coord in contour) == True):
                start_matrix[i][j] = 255
    plt.imshow(start_matrix, cmap="gray", vmin=0, vmax=255)
    plt.show()
    neighbours = calc_neighbours(edges=start_matrix)
    
    square = bfs_square(neighbours=neighbours, coord_start=coord_start)
    return square
def main():


    image = cv2.imread("resources/figures.png")
        
    # convert to gray image
    img_u16 = image.astype(np.uint16)
    gray = (img_u16[:,:,0] + img_u16[:,:,1] + img_u16[:,:,2]) // 3
    gray = gray.astype(np.uint8)
    # filtering an image
    filtered_gray = cv2.medianBlur(gray, 3)

    matrix = np.array([
    [0, 255, 255, 255, 255, 255, 255, 255, 255, 0],
    [255,   0,   0,   0,   0,   0,   0,   0,   0, 255],
    [255,   0, 255, 255, 255, 255, 255, 255,   0, 255],
    [255,   0, 255,   0,   0,   0,   0, 255,   0, 255],
    [255,   0, 255,   0, 255, 255,   0, 255,   0, 255],
    [255,   0, 255,   0, 255, 255,   0, 255,   0, 255],
    [255,   0, 255,   0,   0,   0,   0, 255,   0, 255],
    [255,   0, 0, 255, 255, 255, 255, 255,   0, 255],
    [255,   0,   0,   0,   0,   0,   0,   0,   0, 255],
    [0, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    ])
    list__ = get_contours(gray)
    for row in (list__):
        print(row)
        print('\n')
    # create an image
    square = area(list__[0])
    print(square)


    image = cv2.imread('resources/figures.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 250, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntr = np.array(list__[0])
    mask = np.zeros(gray.shape, dtype=np.uint16)
    cv2.drawContours(mask, [cntr], -1, 255, thickness=cv2.FILLED)
    pixel_count = np.sum(mask == 255)
    print(pixel_count)

if __name__ == "__main__":
    main()