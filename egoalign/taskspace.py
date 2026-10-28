"""Stage 3: map camera-frame hand landmarks to a 6D end-effector trajectory."""

import dataclasses

import numpy as np

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP = 0, 4, 5, 8
MIDDLE_MCP, PINKY_MCP = 9, 17

# Camera (x right, y down, z forward) to robot base (x forward, y left, z up).
CAM_TO_ROBOT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# Hand frame to gripper frame: fingers close along the palm normal.
HAND_TO_GRIPPER = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


@dataclasses.dataclass
class Trajectory:
    """End-effector trajectory in the robot base frame.

    Attributes:
        poses: (T, 6) x, y, z [m], roll, pitch, yaw [rad].
        gripper: (T,) True when closed.
        fps: Frame rate.
    """

    poses: np.ndarray
    gripper: np.ndarray
    fps: float

    def __len__(self):
        """Number of frames."""
        return len(self.poses)

    def __getitem__(self, s):
        """Slices frames; `s` must be a slice."""
        return Trajectory(self.poses[s], self.gripper[s], self.fps)


def euler_to_matrix(rpy):
    """Converts (..., 3) roll-pitch-yaw to (..., 3, 3) as Rz @ Ry @ Rx."""
    r, p, y = np.moveaxis(np.asarray(rpy, dtype=float), -1, 0)
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p),
                              np.cos(y), np.sin(y))
    return np.stack([
        np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
        np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
        np.stack([-sp, cp * sr, cp * cr], -1),
    ], -2)


def matrix_to_euler(rot):
    """Inverse of `euler_to_matrix`; at gimbal lock roll is set to 0."""
    pitch = np.arcsin(np.clip(-rot[..., 2, 0], -1.0, 1.0))
    locked = np.abs(rot[..., 2, 0]) > 1.0 - 1e-9
    roll = np.where(locked, 0.0, np.arctan2(rot[..., 2, 1], rot[..., 2, 2]))
    yaw = np.where(locked, np.arctan2(-rot[..., 0, 1], rot[..., 1, 1]),
                   np.arctan2(rot[..., 1, 0], rot[..., 0, 0]))
    return np.stack([roll, pitch, yaw], -1)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def hand_frame(landmarks, left=False):
    """Returns (T, 3, 3) hand rotations; z is wrist->knuckles (approach)."""
    z = _unit(landmarks[:, MIDDLE_MCP] - landmarks[:, WRIST])
    across = landmarks[:, PINKY_MCP] - landmarks[:, INDEX_MCP]
    if left:
        across = -across
    x = _unit(np.cross(across, z))
    y = np.cross(z, x)
    return np.stack([x, y, z], -1)


def fill_missing(landmarks):
    """Linearly interpolates frames containing any NaN (no hand detected).

    Raises:
        ValueError: If no frame has a hand.
    """
    seen = ~np.isnan(landmarks).any(axis=(1, 2))
    if not seen.any():
        raise ValueError('no hand detected in any frame')
    t = np.arange(len(landmarks))
    flat = landmarks.reshape(len(landmarks), -1)
    filled = np.stack([np.interp(t, t[seen], col[seen]) for col in flat.T], -1)
    return filled.reshape(landmarks.shape)


def smooth(x, window):
    """Centered moving average along axis 0 with edge padding."""
    if window <= 1:
        return x
    pad = window // 2
    padded = np.pad(x, [(pad, window - 1 - pad)] + [(0, 0)] * (x.ndim - 1),
                    mode='edge')
    kernel = np.ones(window) / window
    flat = padded.reshape(len(padded), -1)
    out = np.stack([np.convolve(c, kernel, mode='valid') for c in flat.T], -1)
    return out.reshape(x.shape)


def gripper_states(pinch, close_below, open_above):
    """Thresholds pinch distance with hysteresis; True means closed."""
    closed, out = False, []
    for d in pinch:
        closed = d < close_below if not closed else d <= open_above
        out.append(closed)
    return np.array(out)


def to_trajectory(landmarks, fps, camera_rpy=(0.0, 0.0, 0.0),
                  home=(0.45, 0.0, 0.25), grip_threshold=0.04, window=5,
                  left=False, extrinsics=None):
    """Maps camera-frame landmarks to a robot end-effector trajectory.

    Args:
        landmarks: (T, 21, 3) hand landmarks in metres, camera frame; NaN rows
            for frames without a hand.
        fps: Video frame rate.
        camera_rpy: Camera tilt from the canonical forward view [rad], as
            rotations about camera x (right), y (down), z (forward).
            Canonical points are `euler_to_matrix(camera_rpy) @ p_cam`.
        home: Robot-frame position the first wrist sample maps to [m].
        grip_threshold: Thumb-index tip distance below which the gripper
            closes [m]; it reopens above 1.5x this.
        window: Moving-average window in frames for jitter removal.
        left: True for a left hand.
        extrinsics: Optional `register.Extrinsics`. When given, the
            thumb-index midpoint is placed absolutely in the robot frame and
            `camera_rpy` and `home` are ignored.

    Returns:
        A `Trajectory` in the robot base frame.
    """
    pts = smooth(fill_missing(landmarks), window)
    if extrinsics is None:
        to_robot = CAM_TO_ROBOT @ euler_to_matrix(camera_rpy)
        wrist = pts[:, WRIST] @ to_robot.T
        pos = wrist - wrist[0] + np.asarray(home)
    else:
        to_robot = extrinsics.rot
        grasp = (pts[:, THUMB_TIP] + pts[:, INDEX_TIP]) / 2
        pos = extrinsics.to_robot(grasp)
    rpy = matrix_to_euler(to_robot @ hand_frame(pts, left) @ HAND_TO_GRIPPER)
    pinch = np.linalg.norm(pts[:, THUMB_TIP] - pts[:, INDEX_TIP], axis=-1)
    grip = gripper_states(pinch, grip_threshold, 1.5 * grip_threshold)
    return Trajectory(np.hstack([pos, rpy]), grip, fps)
