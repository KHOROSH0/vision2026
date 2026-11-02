# Isosceles triangle orientation examples

Six images match the gray-on-gray style of `resources/figures.png`. Each contains
a circle, square, rectangle, scalene triangle, equilateral triangle, and one
non-equilateral isosceles triangle rotated clockwise by 0, 45, 90, 150, 225, or
300 degrees. Only the target rotates; the distractors stay fixed.

Vertex **1** is the apex where the equal sides meet. Vertices **2** and **3**
follow clockwise in image coordinates. These IDs follow the same physical
corners through rotation, including when the apex points down. Reflection
reverses clockwise order and is not covered by this convention. Equilateral
triangles cannot have a geometrically unique apex and are rejected.

From the repository root:

```powershell
python src/find_triangle.py
python src/find_triangle.py resources/triangles/figures_triangle_150.png --show
python src/generate_triangle_figures.py
python -m unittest discover -s tests -p test_find_triangle.py
```

The detector saves labeled results to `output/triangles/`. It uses Otsu
thresholding for dark figures on a light background, then the existing manual
`connected_components`, `order_contour` (outer boundary walk), and `rdp`.
Unlike the Canny demo, components here include the filled interiors; the boundary
walk follows their exterior pixels. This is intended for separated synthetic
figures, rather than arbitrary photographs or touching objects.

RDP epsilon is 2% of the traced perimeter. Exactly three vertices must remain;
exactly one vertex must have two incident sides equal within 5%. This tolerance
allows rasterization error, but a nearly isosceles triangle may also qualify.

`expected_vertices.json` records the drawn coordinates in ID order. Tests verify
that every image produces exactly one detection and each ID is within three
pixels of its expected corner. Additional tests cover 24 rotations and every
input vertex permutation, plus rejection of other shapes.
