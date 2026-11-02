"""Create reproducible geometric fixtures, with known clockwise vertex IDs."""
import json
from pathlib import Path

import cv2
import numpy as np


RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources" / "triangles"


def triangle_vertices(angle):
    """IDs 1, 2, 3: apex, then clockwise in image coordinates."""
    points = np.array([(0, -100), (65, 65), (-65, 65)], dtype=float)
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]])
    return np.rint(points @ rotation.T + (490, 310)).astype(np.int32)


def main():
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for angle in (0, 45, 90, 150, 225, 300):
        image = np.full((520, 720, 3), 192, dtype=np.uint8)
        color = (128, 128, 128)
        cv2.circle(image, (100, 100), 60, color, -1)
        cv2.rectangle(image, (40, 240), (165, 365), color, -1)
        cv2.rectangle(image, (220, 380), (350, 445), color, -1)
        # Equilateral and scalene triangles must not be assigned apex IDs.
        equilateral = np.rint([(275, 50), (325, 50 + 50*np.sqrt(3)),
                              (225, 50 + 50*np.sqrt(3))]).astype(np.int32)
        scalene = np.array([(470, 35), (665, 60), (600, 145)], np.int32)
        cv2.fillPoly(image, [equilateral, scalene], color)
        vertices = triangle_vertices(angle)
        cv2.fillPoly(image, [vertices], color)
        filename = f"figures_triangle_{angle:03d}.png"
        if not cv2.imwrite(str(RESOURCE_DIR / filename), image):
            raise OSError(f"Cannot write {filename}")
        manifest.append({"image": filename, "clockwise_rotation_degrees": angle,
                         "vertices_1_2_3": vertices.tolist()})
    (RESOURCE_DIR / "expected_vertices.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {len(manifest)} images in {RESOURCE_DIR}")


if __name__ == "__main__":
    main()
