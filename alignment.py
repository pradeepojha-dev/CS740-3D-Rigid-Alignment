"""CS740: two self-contained 3D rigid point-set alignment algorithms.

Each input is an N x 3 array. Rows are corresponding points, not unordered clouds.
Convention: for column vectors y_i = R @ x_i + t.
In NumPy's row-array convention, Y_hat = X @ R.T + t.
Only NumPy matrix primitives are used (no ready-made registration package).
Reference: Eggert, Lorusso, Fisher (1997), Sections 3.1 and 3.3.
"""
from __future__ import annotations

import numpy as np


def _prepare(source: np.ndarray, target: np.ndarray):
    x, y = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("Source and target must be matching N x 3 arrays (N >= 3)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("All point coordinates must be finite")
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    return x, y, xc, yc, xm, ym


def estimate_svd(source: np.ndarray, target: np.ndarray):
    """Estimate the proper rotation R and translation t by SVD.

    Returns (R, t), with R @ R.T = I and det(R) = +1.
    In exact line/point configurations the orientation is not uniquely determined;
    a numerically valid rotation may still be returned.
    """
    _, _, xc, yc, xm, ym = _prepare(source, target)
    H = xc.T @ yc
    U, singular_values, Vt = np.linalg.svd(H)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ correction @ U.T
    t = ym - R @ xm
    return R, t


def _quat_to_rotation(q: np.ndarray):
    """Quaternion q=[w,x,y,z] to an active 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [w*w+x*x-y*y-z*z, 2*(x*y-w*z),       2*(x*z+w*y)],
        [2*(y*x+w*z),       w*w-x*x+y*y-z*z, 2*(y*z-w*x)],
        [2*(z*x-w*y),       2*(z*y+w*x),       w*w-x*x-y*y+z*z],
    ])


def estimate_quaternion(source: np.ndarray, target: np.ndarray):
    """Estimate R,t using the 4x4 symmetric Horn/Eggert quaternion matrix.

    The unit eigenvector of the largest eigenvalue is the optimal quaternion.
    Implements the matrix from Eggert et al. (1997), Eq. (17), with S_ab =
    sum_i (source_i)_a * (target_i)_b.
    """
    _, _, xc, yc, xm, ym = _prepare(source, target)
    H = xc.T @ yc
    sxx, sxy, sxz = H[0]
    syx, syy, syz = H[1]
    szx, szy, szz = H[2]
    K = np.array([
        [sxx+syy+szz, syz-szy,      szx-sxz,      sxy-syx],
        [syz-szy,     sxx-syy-szz,  sxy+syx,      szx+sxz],
        [szx-sxz,     sxy+syx,      syy-sxx-szz,  syz+szy],
        [sxy-syx,     szx+sxz,      syz+szy,      szz-sxx-syy],
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    q = eigenvectors[:, np.argmax(eigenvalues)]
    R = _quat_to_rotation(q)
    t = ym - R @ xm
    return R, t


def transform(source: np.ndarray, R: np.ndarray, t: np.ndarray):
    return np.asarray(source) @ R.T + t


def axis_angle_to_rotation(axis, angle_degrees: float):
    """Construct test-data rotations via the axis-angle / unit quaternion form."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    h = np.deg2rad(angle_degrees) / 2
    return _quat_to_rotation(np.r_[np.cos(h), np.sin(h) * a])


def rotation_error_degrees(estimated: np.ndarray, truth: np.ndarray):
    """Geodesic angle between two rotations, in degrees."""
    relative = estimated @ truth.T
    cos_angle = np.clip((np.trace(relative) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos_angle)))


def alignment_rmse(source: np.ndarray, target: np.ndarray, R: np.ndarray, t: np.ndarray):
    residuals = transform(source, R, t) - target
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def geometry_ratio(source: np.ndarray):
    """second-largest/largest SVD singular value of centered source points."""
    centered = np.asarray(source) - np.mean(source, axis=0)
    singulars = np.linalg.svd(centered, compute_uv=False)
    return float(singulars[1] / singulars[0]) if singulars[0] > 0 else 0.0
