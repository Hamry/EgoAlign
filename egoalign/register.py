"""Camera-to-robot extrinsics from an ArUco marker lying flat on the table.

The marker is axis-aligned with the robot base: seen from above, its
left-to-right direction is robot +x and bottom-to-top is robot +y.
"""

import contextlib
import dataclasses
import pathlib

import cv2
import numpy as np

from egoalign import extract

DICTIONARY = cv2.aruco.DICT_4X4_50


@dataclasses.dataclass
class Extrinsics:
    """Camera pose in the robot base frame (OpenCV camera axes).

    Attributes:
        rot: (3, 3) camera-to-robot rotation.
        pos: (3,) camera position [m].
    """

    rot: np.ndarray
    pos: np.ndarray

    def to_robot(self, points):
        """Maps (..., 3) camera-frame points into the robot frame."""
        return points @ self.rot.T + self.pos


def write_marker(path, marker_id=0, pixels=800):
    """Saves a printable marker PNG with a white quiet zone; returns its path.

    After printing, measure the black square's side for `--marker-size`.
    """
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(DICTIONARY), marker_id, pixels)
    border = pixels // 6  # At least one marker cell of quiet zone.
    marker = cv2.copyMakeBorder(marker, border, border, border, border,
                                cv2.BORDER_CONSTANT, value=255)
    path = pathlib.Path(path)
    if path.suffix.lower() != '.png':
        path = path.with_suffix('.png')
    if not cv2.imwrite(str(path), marker):
        raise OSError(f'cannot write {path}')
    return path


def marker_corners(center_xy, size):
    """(4, 3) robot-frame corners in ArUco order (TL, TR, BR, BL)."""
    h = size / 2
    offsets = np.array([[-h, h], [h, h], [h, -h], [-h, -h]])
    return np.column_stack([np.asarray(center_xy) + offsets, np.zeros(4)])


def register(rgb, fov_deg, marker_xy, marker_size, marker_id=0):
    """Estimates camera extrinsics from one frame showing the marker.

    Args:
        rgb: (H, W, 3) uint8 frame.
        fov_deg: Horizontal field of view [deg].
        marker_xy: Marker centre in the robot base frame [m].
        marker_size: Printed marker side length [m].
        marker_id: ArUco id in DICT_4X4_50.

    Returns:
        `Extrinsics`, or None if the marker isn't visible.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICTIONARY), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or marker_id not in ids.ravel():
        return None
    image_pts = corners[list(ids.ravel()).index(marker_id)].reshape(4, 2)
    h, w = gray.shape
    ok, rvec, tvec = cv2.solvePnP(
        marker_corners(marker_xy, marker_size), image_pts,
        extract.intrinsics(w, h, fov_deg), None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rot = cv2.Rodrigues(rvec)[0]
    return Extrinsics(rot.T, -rot.T @ tvec.ravel())


def register_video(path, fov_deg, marker_xy, marker_size, marker_id=0,
                   max_frames=60):
    """Registers from the first frame of a video that shows the marker.

    Raises:
        ValueError: If no frame among the first `max_frames` shows the marker.
    """
    with contextlib.closing(extract.read_video(path)) as frames:
        for i, frame in enumerate(frames):
            if i == max_frames:
                break
            ext = register(frame, fov_deg, marker_xy, marker_size, marker_id)
            if ext is not None:
                return ext
    raise ValueError(f'ArUco marker {marker_id} not found in {path}')
