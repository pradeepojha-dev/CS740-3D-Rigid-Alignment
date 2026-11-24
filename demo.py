"""A small and explainable 3D alignment example for the viva.

Run: python demo.py
"""
import numpy as np
from alignment import (estimate_svd, estimate_quaternion, axis_angle_to_rotation,
                       transform, rotation_error_degrees, alignment_rmse)

# Think of each row as one labelled dot on a toy object (x, y, z).
points_before = np.array([
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [1., 1., 1.],
    [1.5, 0.2, 0.8],
])

# Move the object in a way that we know: turn 30 degrees around z-axis,
# then translate by +2 in x, -1 in y, +3 in z.
actual_rotation = axis_angle_to_rotation([0, 0, 1], 30)
actual_translation = np.array([2., -1., 3.])
points_after = transform(points_before, actual_rotation, actual_translation)

print('KNOWN movement: 30 degree rotation around z, translation [2, -1, 3]')
print('--- First three pairs of corresponding 3D points ---')
for before, after in zip(points_before[:3], points_after[:3]):
    print(f'{before}  ->  {np.round(after, 4)}')

for name, function in [('SVD', estimate_svd), ('Unit quaternion', estimate_quaternion)]:
    R, t = function(points_before, points_after)
    print(f'\n{name} results:')
    print('Estimated rotation matrix:\n', np.round(R, 5))
    print('Estimated translation:', np.round(t, 5))
    print('Rotation error (degrees):', rotation_error_degrees(R, actual_rotation))
    print('Fitting RMSE:', alignment_rmse(points_before, points_after, R, t))
    print('Determinant of R:', np.linalg.det(R))
