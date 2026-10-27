"""Find non-equilateral isosceles triangles using manual components and RDP."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from connected_components import connected_components
from contour_simplification import order_contour, rdp


ROOT = Path(__file__).resolve().parent.parent


def identify_vertices(polygon, side_tolerance=0.05, min_area=100):
    """Return apex first and remaining vertices clockwise, or None.

    Exactly one corner must join two approximately equal sides. Equilateral
    triangles are rejected because geometry cannot distinguish their vertices.
    side_tolerance is the allowed relative difference between equal sides.
    """
    points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(points) != 3:
        return None
    a, b = points[1] - points[0], points[2] - points[0]
    signed_double_area = a[0]*b[1] - a[1]*b[0]
    if abs(signed_double_area) < 2*min_area:
        return None

    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    candidates = []
    for vertex in range(3):
        incoming, outgoing = lengths[(vertex - 1) % 3], lengths[vertex]
        if abs(incoming - outgoing) <= side_tolerance * max(incoming, outgoing):
            candidates.append(vertex)
    if len(candidates) != 1:
        return None

    points = np.roll(points, -candidates[0], axis=0)
    # Positive signed area means clockwise on an image, whose y axis goes down.
    if signed_double_area < 0:
        points = points[[0, 2, 1]]
    return points


def find_triangles(image, epsilon_ratio=0.02, side_tolerance=0.05):
    """Return arrays with rows corresponding to vertex IDs 1, 2 and 3."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Flat dark figures on a light background: segment their interiors rather
    # than relying on unbroken Canny edges at rasterized, sloping corners.
    _, foreground = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    components, _ = connected_components(foreground)
    triangles = []
    for component in components:
        if len(component) < 3:
            continue
        try:
            contour = order_contour(component)
        except ValueError:
            continue  # Not a traceable closed boundary.
        points = np.asarray(contour, dtype=float)
        perimeter = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum()
        polygon = rdp(contour, epsilon=epsilon_ratio * perimeter)
        vertices = identify_vertices(polygon, side_tolerance)
        if vertices is not None:
            triangles.append(vertices)
    return triangles


def annotate_triangles(image, triangles):
    result = image.copy()
    for vertices in triangles:
        points = np.rint(vertices).astype(np.int32)
        cv2.polylines(result, [points], True, (0, 150, 0), 2)
        center = vertices.mean(axis=0)
        for vertex_id, point in enumerate(points, start=1):
            direction = point - center
            direction /= np.linalg.norm(direction)
            text_center = point + direction * 23
            text = str(vertex_id)
            (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            origin = tuple(np.rint(text_center + (-width/2, height/2)).astype(int))
            cv2.circle(result, tuple(point), 4, (0, 0, 220), -1)
            cv2.putText(result, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 220), 2, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path,
                        help="Input images; default: all six rotation examples")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "triangles")
    parser.add_argument("--show", action="store_true", help="Also display each result")
    args = parser.parse_args()
    paths = args.images or sorted((ROOT / "resources" / "triangles").glob("figures_triangle_*.png"))
    if not paths:
        parser.error("No images found; run src/generate_triangle_figures.py first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        triangles = find_triangles(image)
        result = annotate_triangles(image, triangles)
        destination = args.output_dir / f"{path.stem}_annotated.png"
        if not cv2.imwrite(str(destination), result):
            raise OSError(f"Cannot write {destination}")
        print(f"{path.name}: {len(triangles)} isosceles triangle(s) -> {destination}")
        for vertices in triangles:
            print("  IDs 1, 2, 3:", vertices.astype(int).tolist())
        if args.show:
            cv2.imshow("Isosceles triangle: clockwise vertex IDs", result)
            cv2.waitKey(0)
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
