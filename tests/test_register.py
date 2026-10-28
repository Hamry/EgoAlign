"""ArUco registration recovers a known camera pose from a synthetic image."""

import cv2
import numpy as np

from egoalign import extract
from egoalign import register
from egoalign import sim

W, H, FOV = 640, 480, 65.0


def _render_marker(cam_pos, cam_rot, center, size):
    """Warps a marker (with white quiet zone) into a synthetic camera view."""
    px, border = 200, 40
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(register.DICTIONARY), 0, px)
    tile = cv2.copyMakeBorder(marker, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=255)
    outer = register.marker_corners(center, size * (px + 2 * border) / px)
    cam = (outer - cam_pos) @ cam_rot
    k = extract.intrinsics(W, H, FOV)
    uv = (cam @ k.T)[:, :2] / cam[:, 2:]
    n = tile.shape[0] - 1
    src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    warp = cv2.getPerspectiveTransform(src, np.float32(uv))
    gray = cv2.warpPerspective(tile, warp, (W, H), borderValue=90)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def test_register_recovers_camera_pose():
    cam_pos = np.array([0.9, 0.3, 0.6])
    cam_rot = sim.look_at(cam_pos, [0.4, 0.0, 0.0])
    image = _render_marker(cam_pos, cam_rot, (0.4, 0.0), 0.1)
    ext = register.register(image, FOV, (0.4, 0.0), 0.1)
    assert ext is not None
    np.testing.assert_allclose(ext.pos, cam_pos, atol=0.01)
    np.testing.assert_allclose(ext.rot, cam_rot, atol=0.02)


def test_register_without_marker_returns_none():
    assert register.register(np.zeros((H, W, 3), np.uint8), FOV,
                             (0, 0), 0.1) is None
