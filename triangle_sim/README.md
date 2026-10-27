# Planar robot with a flange camera

From the repository root:

```powershell
python -m pip install -r triangle_sim/requirements.txt
python triangle_sim/run.py
```

The live OpenCV window shows the 3D scene on the left and the flange camera on
the right. IDs are **1 = apex**, **2 and 3 = clockwise**. Press **Q** or **Esc** to
quit, or **S** to save `triangle_sim/output/scene.png`. Closing the window exits
too. Optionally use `--gui` to open PyBullet's interactive 3D view alongside it.
The default uses PyBullet DIRECT with its CPU TinyRenderer, so only the OpenCV
window is required. No external robot models or downloads are needed.

## Geometry and motion

- A planar 3R arm has three revolute joints around world Z, link lengths
  0.45, 0.35 and 0.18 m, and its base at Z = 0.65 m.
- The camera is a fixed fourth link at the flange, offset 0.08 m downward.
  Its optical axis is local -Z and image up is local +Y. Both vectors are
  transformed using the actual camera link orientation each frame. The camera
  does not independently aim at or track the marker.
- A fixed black isosceles triangle sits on a white horizontal board. A 75-degree
  vertical field of view and small joint excursions keep it within the image.
  Keeping the camera parallel to the board preserves the isosceles geometry;
  arbitrary perspective tilt would require rectification or a different detector.
- `scene.py:joint_angles` defines smooth 12-second periodic joint trajectories.
  These are prescribed kinematics (`resetJointState`), assuming ideal tracking,
  rather than motor control or a dynamics experiment. The live animation follows
  elapsed wall time. There is no visual servoing or image-based feedback.

`run.py` directly imports the latest `find_triangles` and `annotate_triangles`
from `src/find_triangle.py`. Detection uses only rendered RGB pixels: Otsu
thresholding, manual connected components, manual boundary tracing and RDP.
World coordinates and renderer segmentation are not supplied to the detector.
The detector runs on 320x240 images; the display is enlarged for readability.
`--fps 15` sets a display-rate ceiling; actual speed depends on the Python
detector and machine.

## Validate a full cycle without windows

```powershell
python triangle_sim/run.py --headless --frames 48
```

The check samples an entire motion period, requires exactly one detection in
every frame, verifies every vertex ID against independently projected 3D marker
coordinates within 5 pixels, and requires at least 25 pixels of image margin.
It saves `output/camera.png`, `output/scene.png`, and `output/cycle.png` inside
this folder. Ground-truth projection is used only for this check, not detection.
The check exits with an error and saves `failure.png` if visibility or detection
fails. `--output-dir` selects another destination.

Camera API reference: [official PyBullet camera example](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getCameraImageTest.py).
