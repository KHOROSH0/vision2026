"""Live PyBullet flange camera with rotation-stable triangle vertex IDs."""
import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from scene import HEIGHT, PERIOD, WIDTH, TriangleScene

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from find_triangle import annotate_triangles, find_triangles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Also open PyBullet's 3D window")
    parser.add_argument("--headless", action="store_true", help="Validate a full cycle without windows")
    parser.add_argument("--frames", type=int, default=48, help="Samples per headless cycle")
    parser.add_argument("--fps", type=float, default=15, help="Maximum live display rate")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    if args.frames < 4 or args.fps <= 0:
        parser.error("--frames must be at least 4 and --fps must be positive")
    if args.gui and args.headless:
        parser.error("--gui and --headless cannot be combined")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = TriangleScene(gui=args.gui)
    started = time.perf_counter()
    count, worst_error, minimum_margin = 0, 0.0, float("inf")
    samples = []
    try:
        while True:
            frame_started = time.perf_counter()
            t = count * PERIOD / args.frames if args.headless else frame_started - started
            scene.set_time(t)
            image = scene.render()
            triangles = find_triangles(image)
            annotated = annotate_triangles(image, triangles)
            overview = scene.render(overview=True)
            overview = cv2.copyMakeBorder(overview, 25, 0, 0, 0, cv2.BORDER_CONSTANT,
                                          value=(245, 245, 245))
            annotated = cv2.copyMakeBorder(annotated, 25, 0, 0, 0, cv2.BORDER_CONSTANT,
                                           value=(245, 245, 245))
            for tile, text in [(overview, "Planar 3R arm / rigid flange camera"),
                               (annotated, f"Camera: {len(triangles)} triangle(s), t={t:.1f}s")]:
                cv2.putText(tile, text, (7, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.40, (40, 40, 220), 1, cv2.LINE_AA)
                # Separate the title bar from the rendered image.
                cv2.line(tile, (0, 24), (tile.shape[1]-1, 24), (65, 65, 65), 1)
            overview = cv2.copyMakeBorder(overview, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                          value=(65, 65, 65))
            annotated = cv2.copyMakeBorder(annotated, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                           value=(65, 65, 65))
            display = np.hstack((overview, annotated))
            if args.headless:
                expected = scene.projected_vertices()
                margin = float(np.min(np.column_stack(
                    (expected[:, 0], WIDTH-expected[:, 0], expected[:, 1], HEIGHT-expected[:, 1]))))
                minimum_margin = min(minimum_margin, margin)
                if margin < 25 or len(triangles) != 1:
                    cv2.imwrite(str(args.output_dir / "failure.png"), display)
                    raise AssertionError(f"t={t:.2f}: margin={margin:.1f}, detections={len(triangles)}")
                error = float(np.linalg.norm(triangles[0] - expected, axis=1).max())
                worst_error = max(worst_error, error)
                if error > 5:
                    cv2.imwrite(str(args.output_dir / "failure.png"), display)
                    raise AssertionError(f"t={t:.2f}: incorrect vertex IDs or error={error:.1f}px")
                if count % max(1, args.frames//6) == 0 and len(samples) < 6:
                    samples.append(display)
                if count == 0:
                    cv2.imwrite(str(args.output_dir / "camera.png"), annotated)
                    cv2.imwrite(str(args.output_dir / "scene.png"), display)
                count += 1
                if count == args.frames:
                    break
            else:
                cv2.imshow("Triangle simulation | Q/Esc: quit | S: save", cv2.resize(
                    display, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))
                delay = max(1, round(1000*(1/args.fps - (time.perf_counter()-frame_started))))
                key = cv2.waitKey(delay) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(
                        "Triangle simulation | Q/Esc: quit | S: save", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("s"):
                    cv2.imwrite(str(args.output_dir / "scene.png"), display)
                count += 1
        if args.headless:
            cv2.imwrite(str(args.output_dir / "cycle.png"), np.vstack(samples))
            print(f"PASS: {count} phases over {PERIOD:g}s; exactly one triangle each frame; "
                  f"max vertex error {worst_error:.2f}px; min FoV margin {minimum_margin:.1f}px")
    finally:
        scene.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
