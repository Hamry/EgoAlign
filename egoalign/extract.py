"""Stage 1: video decoding and 3D hand landmark extraction with MediaPipe."""

import os
import pathlib
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
import numpy as np

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
             'hand_landmarker/float16/1/hand_landmarker.task')


def model_path():
    """Returns the cached hand landmarker model, downloading it if missing."""
    cache = pathlib.Path(os.environ.get(
        'EGOALIGN_CACHE', pathlib.Path.home() / '.cache' / 'egoalign'))
    path = cache / 'hand_landmarker.task'
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.part')
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(path)
    return path


def read_video(path):
    """Yields RGB frames from a video file.

    Args:
        path: Video file.

    Yields:
        (H, W, 3) uint8 RGB frames.

    Raises:
        ValueError: If the file cannot be opened.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'cannot open video: {path}')
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def video_fps(path, default=30.0):
    """Returns the container frame rate.

    Args:
        path: Video file.
        default: Rate used when the container reports none or a bogus one.

    Raises:
        ValueError: If the file cannot be opened.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if 0 < fps <= 240 else default


def intrinsics(width, height, fov_deg):
    """Pinhole matrix with principal point at the image centre."""
    f = width / 2 / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def lift(image_lm, world_lm, width, height, fov_deg):
    """Places metric hand landmarks in the camera frame via pinhole fitting.

    Solves the translation t minimising reprojection error of world_lm + t
    against the image landmarks; linear least squares in t.

    Args:
        image_lm: (21, 3) normalized image landmarks (x, y in [0, 1]).
        world_lm: (21, 3) hand-centred metric landmarks, camera-aligned axes.
        width: Image width [px].
        height: Image height [px].
        fov_deg: Horizontal field of view of the camera [deg].

    Returns:
        (21, 3) landmarks in the camera frame [m]; all-NaN if the fit puts
        the hand behind the camera.
    """
    k = intrinsics(width, height, fov_deg)
    f = k[0, 0]
    u = image_lm[:, 0] * width - k[0, 2]
    v = image_lm[:, 1] * height - k[1, 2]
    wx, wy, wz = world_lm.T
    zeros = np.zeros_like(u)
    a = np.vstack([np.column_stack([np.full_like(u, f), zeros, -u]),
                   np.column_stack([zeros, np.full_like(v, f), -v])])
    b = np.concatenate([u * wz - f * wx, v * wz - f * wy])
    t = np.linalg.lstsq(a, b, rcond=None)[0]
    if (world_lm[:, 2] + t[2]).min() <= 0:
        return np.full_like(world_lm, np.nan)
    return world_lm + t


class HandTracker:
    """Per-frame single-hand tracker producing camera-frame landmarks."""

    def __init__(self, fps, fov_deg=65.0):
        """Creates the tracker.

        Args:
            fps: Video frame rate, for timestamps.
            fov_deg: Horizontal camera field of view [deg].
        """
        options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(
                model_asset_path=str(model_path())),
            running_mode=vision.RunningMode.VIDEO, num_hands=1)
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._fps = fps
        self._fov = fov_deg
        self._frame = 0
        self.left_votes = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Releases the native landmarker."""
        self._landmarker.close()

    def process(self, rgb):
        """Returns (21, 3) camera-frame landmarks for the next frame.

        Args:
            rgb: (H, W, 3) uint8 RGB frame.

        Returns:
            Landmarks [m], or all-NaN if no hand was found.
        """
        ts_ms = int(self._frame * 1000 / self._fps)
        self._frame += 1
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.hand_landmarks:
            return np.full((21, 3), np.nan)
        # ponytail: assumes mirrored-selfie labels (unverified on Tasks API).
        label = result.handedness[0][0].category_name
        self.left_votes += 1 if label == 'Right' else -1
        img = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])
        world = np.array([[p.x, p.y, p.z]
                          for p in result.hand_world_landmarks[0]])
        h, w = rgb.shape[:2]
        return lift(img, world, w, h, self._fov)
