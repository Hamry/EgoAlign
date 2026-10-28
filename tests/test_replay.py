"""Physics replay: a scripted top-down pick lifts the mug; a miss doesn't."""

import numpy as np

from egoalign import replay
from egoalign import sim
from egoalign import taskspace

MUG = (0.5, 0.1)
DOWN = (np.pi, 0.0, 0.0)


def _segment(a, b, frames=30):
    t = np.linspace(0, 1, frames, endpoint=False)[:, None]
    return np.r_[a] + (np.r_[b] - np.r_[a]) * t


def _pick(close):
    """Approach, descend onto the mug, (maybe) close, lift."""
    grasp = (*MUG, 0.07, *DOWN)
    reach = np.vstack([_segment((0.45, 0.1, 0.3, *DOWN), (*MUG, 0.2, *DOWN)),
                       _segment((*MUG, 0.2, *DOWN), grasp)])
    lift = np.vstack([_segment(grasp, (*MUG, 0.3, *DOWN)),
                      [(*MUG, 0.3, *DOWN)]])
    grip = np.r_[np.zeros(len(reach), bool), np.full(len(lift), close)]
    return taskspace.Trajectory(np.vstack([reach, lift]), grip, 30.0)


def _run(close):
    model = sim.build(mug_xy=MUG)
    traj = _pick(close)
    joints = replay.joint_path(model, traj)
    assert joints is not None
    joints, grip = replay.hold_on_grip_change(joints, traj.gripper, 15)
    return replay.replay(model, joints, grip, traj.fps, width=64, height=64)


def test_scripted_pick_succeeds():
    rollout = _run(close=True)
    assert rollout.success
    n = len(rollout.images)
    assert rollout.images.shape == (n, 64, 64, 3)
    assert rollout.states.shape == rollout.actions.shape == (n, 8)


def test_open_gripper_fails():
    assert not _run(close=False).success


def test_hold_on_grip_change_repeats_toggle_frame():
    joints = np.arange(4)[:, None] * np.ones(7)
    grip = np.array([False, False, True, True])
    j, g = replay.hold_on_grip_change(joints, grip, 2)
    assert j[:, 0].tolist() == [0, 1, 2, 2, 2, 3]
    assert g.tolist() == [False, False, True, True, True, True]
