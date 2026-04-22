import math
import numpy as np

from PyQt6.QtGui import QTransform

# convert 3x3 numpy homography to QTransform
# OpenCV: p' = H @ p (column vectors)
# Qt: p' = p * M (row vectors), so Qt matrix = H transposed in terms of index mapping
# mapping: QTransform(m11,m12,m13, m21,m22,m23, dx,dy,m33)
# where x' = m11*x + m21*y + dx,  y' = m12*x + m22*y + dy
# matching OpenCV: m11=H[0,0], m21=H[0,1], dx=H[0,2], m12=H[1,0], ...
def h_to_qt(H):
    return QTransform(
        H[0,0], H[1,0], H[2,0],
        H[0,1], H[1,1], H[2,1],
        H[0,2], H[1,2], H[2,2]
    )


# build similarity homography from tx, ty, uniform scale, rotation degrees
def build_similarity_H(tx, ty, scale, rot_deg):
    r = math.radians(rot_deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([
        [scale * c, -scale * s, tx],
        [scale * s,  scale * c, ty],
        [0, 0, 1]
    ], dtype=np.float64)


# decompose H into approximate (tx, ty, scale, rotation)
# exact only for similarity transforms; perspective H returns best approximation
def decompose_H(H):
    tx = float(H[0, 2])
    ty = float(H[1, 2])
    scale = float(math.sqrt(H[0, 0] **2 + H[1, 0]** 2))
    rot = float(math.degrees(math.atan2(H[1, 0], H[0, 0])))
    return tx, ty, scale, rot


# applies homography H to 2D points (N,2) and returns transformed points (N,2)
def apply_H(H, pts):
    pts = np.asarray(pts)
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])
    out = (H @ pts_h.T).T
    out /= out[:, 2:3]
    return out[:, :2]