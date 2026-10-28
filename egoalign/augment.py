"""Turns one demo into many by moving the mug and retargeting the reach."""

import numpy as np

from egoalign import replay
from egoalign import sim
from egoalign import taskspace


def grasp_frame(traj):
    """Index of the first closed-gripper frame.

    Raises:
        ValueError: If the gripper never closes.
    """
    closed = np.flatnonzero(traj.gripper)
    if not len(closed):
        raise ValueError('demo never closes the gripper')
    return int(closed[0])


def retarget(traj, delta_xy):
    """Shifts the trajectory by `delta_xy`, ramping in until the grasp.

    The start pose is unchanged; from the grasp on, the full shift applies.
    """
    g = grasp_frame(traj)
    t = np.clip(np.arange(len(traj)) / g, 0.0, 1.0) if g else np.ones(len(traj))
    ramp = t * t * (3 - 2 * t)  # Smoothstep.
    poses = traj.poses.copy()
    poses[:, :2] += ramp[:, None] * np.asarray(delta_xy)
    return taskspace.Trajectory(poses, traj.gripper.copy(), traj.fps)


def generate(traj, n, scene_kwargs=None, radius=0.08, seed=0, hold_frames=15,
             max_attempts=None, width=256, height=256):
    """Yields up to `n` successful (mug_xy, Rollout) variations of a demo.

    The demo's mug is assumed to sit under its grasp point.

    Args:
        traj: Robot-frame `taskspace.Trajectory` of one pick.
        n: Successful episodes wanted.
        scene_kwargs: Extra `sim.build` arguments (camera pose, FOV).
        radius: Half-width of the square of mug offsets [m].
        seed: RNG seed.
        hold_frames: Frames held at each gripper toggle.
        max_attempts: Attempts before giving up; defaults to 3 * n.
        width: Render width [px].
        height: Render height [px].
    """
    rng = np.random.default_rng(seed)
    src = traj.poses[grasp_frame(traj), :2]
    found = 0
    for i in range(3 * n if max_attempts is None else max_attempts):
        if found == n:
            return
        delta = np.zeros(2) if i == 0 else rng.uniform(-radius, radius, 2)
        model = sim.build(mug_xy=tuple(src + delta), width=width,
                          height=height, **(scene_kwargs or {}))
        shifted = retarget(traj, delta)
        joints = replay.joint_path(model, shifted)
        if joints is None:
            continue
        joints, grip = replay.hold_on_grip_change(joints, shifted.gripper,
                                                  hold_frames)
        rollout = replay.replay(model, joints, grip, traj.fps, width, height)
        if rollout.success:
            found += 1
            yield src + delta, rollout
