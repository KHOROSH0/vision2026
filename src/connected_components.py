import cv2
import numpy as np


def connected_components(binary_image, connectivity=8):
    """
    Find connected components in a binary image manually.

    Parameters
    ----------
    binary_image : np.ndarray
        uint8 binary image: 0 = background, non-zero = foreground

    connectivity : int
        4 or 8

    Returns
    -------
    components : list
        Each component is a list of (x, y) pixel coordinates.

    labels : np.ndarray
        int32 label image:
            0 = background
            1 = first component
            2 = second component
            ...
    """

    height, width = binary_image.shape

    # Initially every pixel is unlabeled.
    labels = np.zeros((height, width), dtype=np.int32)

    if connectivity == 4:
        neighbors = [
            (-1,  0),
            ( 1,  0),
            ( 0, -1),
            ( 0,  1)
        ]

    elif connectivity == 8:
        neighbors = [
            (-1, -1), ( 0, -1), ( 1, -1),
            (-1,  0),           ( 1,  0),
            (-1,  1), ( 0,  1), ( 1,  1)
        ]

    else:
        raise ValueError("connectivity must be 4 or 8")

    components = []
    current_label = 0

    # Scan the entire image.
    for y in range(height):
        for x in range(width):

            # Skip background.
            if binary_image[y, x] == 0:
                continue

            # Skip pixels that already belong to a component.
            if labels[y, x] != 0:
                continue

            # We found a new component.
            current_label += 1

            component = []

            # Stack for DFS.
            stack = [(x, y)]

            # Mark immediately so that we don't add the same
            # pixel to the stack multiple times.
            labels[y, x] = current_label

            while stack:

                px, py = stack.pop()

                component.append((px, py))

                # Look at neighboring pixels.
                for dx, dy in neighbors:

                    nx = px + dx
                    ny = py + dy

                    # Check image boundaries.
                    if nx < 0 or nx >= width:
                        continue

                    if ny < 0 or ny >= height:
                        continue

                    # Background pixel.
                    if binary_image[ny, nx] == 0:
                        continue

                    # Already visited.
                    if labels[ny, nx] != 0:
                        continue

                    # This pixel belongs to the same component.
                    labels[ny, nx] = current_label
                    stack.append((nx, ny))

            components.append(component)

    return components, labels

def component_area(component, image_shape):
    """
    Calculate area enclosed by a closed connected component.

    component:
        list of (x, y) boundary pixels

    image_shape:
        (height, width) or image.shape

    Returns:
        area - number of pixels inside or on the boundary
    """

    height, width = image_shape[:2]

    # 0 = unknown
    # 1 = boundary
    # 2 = outside
    labels = np.zeros((height, width), dtype=np.uint8)

    # Mark boundary pixels
    for x, y in component:
        labels[y, x] = 1

    # Flood fill the outside.
    stack = [(0, 0)]

    while stack:
        x, y = stack.pop()

        # Outside image
        if x < 0 or x >= width:
            continue
        if y < 0 or y >= height:
            continue

        # Boundary or already visited
        if labels[y, x] != 0:
            continue

        labels[y, x] = 2

        # 4-connected flood fill
        stack.append((x - 1, y))
        stack.append((x + 1, y))
        stack.append((x, y - 1))
        stack.append((x, y + 1))

    # Everything that is NOT outside is either
    # boundary or interior.
    area = 0

    for y in range(height):
        for x in range(width):
            if labels[y, x] != 2:
                area += 1

    return area


# **
# **
# p = 4
# a = 1
# s_t = 1
# s_f = 4

# ***
# * *
# ***
# p = 8
# s_f = 9
# a = 2
# s_t = 4

# ****
# *  *
# *  *
# ****
# p = 12
# s_f = 16
# a = 3
# s_t = 9
