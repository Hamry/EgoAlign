"""Physics replay of a trajectory on the sim Panda, rendering the POV camera.

Action convention (shared with closed-loop eval): 7 joint targets + gripper
command, 1 = closed. Each observation is taken before its action is applied.
"""

import dataclasses

import mujoco
import numpy as np

from egoalign import robot
from egoalign import sim

GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
LIFT_SUCCESS = 0.1


@dataclasses.dataclass
class Rollout:
    """One rendered episode.

    Attributes:
        images: (T, H, W, 3) uint8 POV frames, each taken before its action.
        states: (T, 8) float32 joint angles + gripper width [m].
        actions: (T, 8) float32 joint targets + gripper command (1 closed).
        success: True if the mug rose at least LIFT_SUCCESS above its rest.
    """

    images: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    success: bool


def joint_path(model, traj, iters=100):
    """Solves IK along a trajectory on `model`; returns (T, 7) or None."""
    panda = robot.Panda(model)
    q, path = robot.READY, []
    for pose in traj.poses:
        q, ok = panda.solve(pose, q, iters=iters)
        if not ok:
            return None
        path.append(q)
    return np.array(path)


def hold_on_grip_change(joints, gripper, frames):
    """Repeats the frame at each gripper toggle so the fingers can settle."""
    toggles = np.flatnonzero(np.diff(gripper.astype(int))) + 1
    idx = np.insert(np.arange(len(joints)), np.repeat(toggles, frames),
                    np.repeat(toggles, frames))
    return joints[idx], gripper[idx]


def control_substeps(model, fps):
    """Physics steps per control step; retunes the timestep to hit `fps`."""
    substeps = max(1, round(1 / (fps * model.opt.timestep)))
    model.opt.timestep = 1 / (fps * substeps)
    return substeps


def reset(model, q0, settle_s=0.5):
    """Returns sim data settled at joint configuration `q0`, gripper open."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:7] = q0
    apply_action(data, np.r_[q0, 0.0])
    mujoco.mj_step(model, data, round(settle_s / model.opt.timestep))
    return data


def apply_action(data, action):
    """Writes an 8-D action (joint targets, gripper 1 = closed) to ctrl."""
    data.ctrl[:7] = action[:7]
    data.ctrl[7] = GRIP_CLOSED if action[7] >= 0.5 else GRIP_OPEN


def observe(renderer, data):
    """Returns (POV image, 8-D state) for the current sim state."""
    renderer.update_scene(data, 'pov')
    state = np.r_[data.qpos[:7], data.qpos[7] + data.qpos[8]]
    return renderer.render(), state.astype(np.float32)


def replay(model, joints, gripper, fps, width=256, height=256):
    """Drives the arm through joint targets with physics and renders it.

    Args:
        model: Scene from `sim.build`; its offscreen buffer must fit the size.
        joints: (T, 7) joint targets.
        gripper: (T,) True to close.
        fps: Control rate; the physics timestep is retuned to divide it.
        width: Render width [px].
        height: Render height [px].

    Returns:
        A `Rollout`.

    Raises:
        ValueError: On empty or mismatched inputs.
    """
    if not len(joints) or len(joints) != len(gripper):
        raise ValueError('joints and gripper must be non-empty, same length')
    substeps = control_substeps(model, fps)
    data = reset(model, joints[0])
    images, states, actions = [], [], []
    with mujoco.Renderer(model, height, width) as renderer:
        for q, closed in zip(joints, gripper):
            image, state = observe(renderer, data)
            action = np.r_[q, float(closed)].astype(np.float32)
            images.append(image)
            states.append(state)
            actions.append(action)
            apply_action(data, action)
            mujoco.mj_step(model, data, substeps)
    return Rollout(np.array(images), np.array(states), np.array(actions),
                   bool(sim.mug_height(data) >= LIFT_SUCCESS))
