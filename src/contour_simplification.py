import numpy as np


def order_contour(component):
    """Order a one-pixel-wide, closed contour with no branches.

    Input is one (x, y) component from connected_components. Walk clockwise
    through eight-neighbor pixels until returning to the start. The returned
    list has no repeated closing point. Assumes a simple, non-touching loop;
    this function does not validate those assumptions.
    """
    pixels = set(map(tuple, component))
    if not pixels:
        return []

    # Clockwise neighbors, starting west, in image coordinates (y grows down).
    neighbors = [(-1, 0), (-1, -1), (0, -1), (1, -1),
                 (1, 0), (1, 1), (0, 1), (-1, 1)]
    start = min(pixels, key=lambda point: (point[1], point[0]))
    current = start
    backtrack = (start[0] - 1, start[1])
    contour = []

    # A simple loop visits at most every component pixel before closing.
    for _ in range(len(pixels)):
        contour.append(current)
        offset = (backtrack[0] - current[0], backtrack[1] - current[1])
        begin = neighbors.index(offset)

        for step in range(1, 9):
            index = (begin + step) % 8
            dx, dy = neighbors[index]
            candidate = (current[0] + dx, current[1] + dy)
            if candidate in pixels:
                break
        else:
            raise ValueError("Contour pixel has no neighboring pixel")

        if candidate == start:
            return contour

        dx, dy = neighbors[(index - 1) % 8]
        backtrack = (current[0] + dx, current[1] + dy)
        current = candidate

    raise ValueError("Contour did not close within the component's pixel count")


def rdp(contour, epsilon):
    """Manually simplify an ordered, closed (x, y) contour with RDP.

    epsilon is the maximum distance in pixels from a point to its replacement
    segment. Closed contours are split into two open arcs before simplifying,
    avoiding a zero-length segment between identical start/end points.
    Returns a list of (x, y) vertices without a repeated closing point.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(points) > 1 and np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) <= 2:
        return [tuple(point) for point in points]

    def simplify_arc(arc):
        # An explicit stack avoids Python's recursion limit on long contours.
        keep = {0, len(arc) - 1}
        stack = [(0, len(arc) - 1)]
        while stack:
            first, last = stack.pop()
            if last - first <= 1:
                continue
            segment = arc[last] - arc[first]
            offsets = arc[first + 1:last] - arc[first]
            length_squared = np.dot(segment, segment)
            if length_squared == 0:
                distances = np.linalg.norm(offsets, axis=1)
            else:
                projection = np.clip(offsets @ segment / length_squared, 0, 1)
                distances = np.linalg.norm(
                    offsets - projection[:, None] * segment, axis=1
                )
            farthest = int(np.argmax(distances))
            if distances[farthest] > epsilon:
                split = first + 1 + farthest
                keep.add(split)
                stack.extend([(first, split), (split, last)])
        return arc[sorted(keep)]

    split = int(np.argmax(np.linalg.norm(points - points[0], axis=1)))
    if split == 0:
        return [tuple(points[0])]
    first_arc = simplify_arc(points[:split + 1])
    second_arc = simplify_arc(np.vstack((points[split:], points[:1])))
    simplified = np.vstack((first_arc[:-1], second_arc[:-1]))
    return [tuple(point) for point in simplified]
