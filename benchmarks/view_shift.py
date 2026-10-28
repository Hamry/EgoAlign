"""KPI 2: wrist-trajectory MAE between a canonical and an off-angle camera.

Synthesizes one hand motion, renders it to two pinhole cameras with landmark
noise, and runs lifting plus task-space mapping on each view. Metric hand
shape gets only i.i.d. noise, so this is optimistic versus real MediaPipe.
"""

import numpy as np

from egoalign import extract
from egoalign import taskspace

WIDTH, HEIGHT, FOV = 1280, 720, 65.0
FRAMES = 90
FPS = 30.0
TARGET = np.array([0.0, 0.1, 0.5])


def hand_template():
    """(21, 3) hand in its own frame: wrist at origin, fingers along +z."""
    lm = np.zeros((21, 3))
    for finger in range(5):
        for joint in range(4):
            x = (finger - 2) * 0.02 - (0.03 if finger == 0 else 0.0)
            lm[1 + 4 * finger + joint] = [x, 0.0, 0.03 + 0.025 * (joint + 1)]
    return lm


def motion():
    """(T, 21, 3) reach-grasp-place motion in the canonical camera frame."""
    t = np.linspace(0, 1, FRAMES)
    wrist = TARGET + np.stack([0.2 * t - 0.1, 0.08 * np.sin(np.pi * t),
                               0.05 * t], -1)
    rots = taskspace.euler_to_matrix(
        np.stack([np.pi / 2 + 0.3 * t, 0.2 * t, 0.1 * np.ones_like(t)], -1))
    return np.einsum('tij,kj->tki', rots, hand_template()) + wrist[:, None]


def render(points, rot, rng, px_noise=2.0, world_noise=0.003):
    """Projects canonical points into a camera aimed at TARGET, with noise.

    Args:
        points: (T, 21, 3) canonical-frame landmarks.
        rot: Camera rotation, canonical-from-camera.
        rng: numpy Generator.
        px_noise: Image landmark noise std [px].
        world_noise: Metric landmark noise std [m].

    Returns:
        Per-frame (image, world) landmark pairs as MediaPipe would give them.
    """
    centre = TARGET - rot @ np.array([0.0, 0.0, TARGET[2]])
    f = WIDTH / 2 / np.tan(np.radians(FOV) / 2)
    out = []
    for p in points:
        cam = (p - centre) @ rot
        uv = f * cam[:, :2] / cam[:, 2:] + rng.normal(0, px_noise, (21, 2))
        image = np.column_stack([(uv[:, 0] + WIDTH / 2) / WIDTH,
                                 (uv[:, 1] + HEIGHT / 2) / HEIGHT,
                                 np.zeros(21)])
        world = cam - cam.mean(0) + rng.normal(0, world_noise, (21, 3))
        out.append((image, world))
    return out


def observe(points, rot, rng):
    """Renders one view and lifts it back to (T, 21, 3) camera-frame points."""
    return np.array([extract.lift(image, world, WIDTH, HEIGHT, FOV)
                     for image, world in render(points, rot, rng)])


def mae_cm(rpy, correct, seed=0):
    """Translational MAE [cm] between canonical and off-angle trajectories.

    Args:
        rpy: Off-angle camera rotation about camera x, y, z [rad].
        correct: Pass `rpy` to the mapping as the known camera tilt.
        seed: Noise seed.
    """
    rng = np.random.default_rng(seed)
    points = motion()
    ref = taskspace.to_trajectory(observe(points, np.eye(3), rng), FPS)
    rot = taskspace.euler_to_matrix(rpy)
    off = taskspace.to_trajectory(observe(points, rot, rng), FPS,
                                  camera_rpy=rpy if correct else (0, 0, 0))
    return 100 * np.linalg.norm(off.poses[:, :3] - ref.poses[:, :3],
                                axis=-1).mean()


def main():
    print(f'{"offset":<24}{"corrected":>12}{"uncorrected":>14}')
    cases = [('baseline', 0, 0)] + [
        (f'{deg} deg {axis}', deg, i) for deg in (10, 20, 30)
        for i, axis in enumerate(('pitch (cam x)', 'yaw (cam y)',
                                  'roll (cam z)'))]
    for label, deg, i in cases:
        rpy = np.radians(np.eye(3)[i] * deg)
        print(f'{label:<24}{mae_cm(rpy, True):>9.2f} cm'
              f'{mae_cm(rpy, False):>11.2f} cm')


if __name__ == '__main__':
    main()
