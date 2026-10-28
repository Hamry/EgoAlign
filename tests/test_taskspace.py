"""Tests for Stage 3 task-space mapping."""

import numpy as np

from egoalign import taskspace


def _hand(wrist, pinch=0.08):
    """Synthetic hand pointing along camera +z, palm facing camera -y."""
    lm = np.zeros((21, 3))
    lm[:] = wrist
    lm[taskspace.MIDDLE_MCP] = wrist + [0, 0, 0.09]
    lm[taskspace.INDEX_MCP] = wrist + [-0.03, 0, 0.08]
    lm[taskspace.PINKY_MCP] = wrist + [0.03, 0, 0.07]
    lm[taskspace.THUMB_TIP] = wrist + [-0.05, 0, 0.1]
    lm[taskspace.INDEX_TIP] = lm[taskspace.THUMB_TIP] + [pinch, 0, 0]
    return lm


def test_euler_roundtrip():
    rpy = np.random.default_rng(0).uniform(-1.4, 1.4, (50, 3))
    np.testing.assert_allclose(
        taskspace.matrix_to_euler(taskspace.euler_to_matrix(rpy)), rpy,
        atol=1e-9)


def test_forward_motion_maps_to_robot_x():
    lm = np.stack([_hand(np.array([0, 0, 0.5 + 0.01 * t])) for t in range(20)])
    traj = taskspace.to_trajectory(lm, fps=30, window=1)
    np.testing.assert_allclose(traj.poses[0, :3], [0.45, 0, 0.25])
    np.testing.assert_allclose(traj.poses[-1, :3], [0.64, 0, 0.25], atol=1e-9)
    # Hand z (approach) points along camera z -> robot +x.
    approach = taskspace.euler_to_matrix(traj.poses[0, 3:])[:, 2]
    np.testing.assert_allclose(approach, [1, 0, 0], atol=1e-9)


def test_camera_tilt_is_undone():
    lm = np.stack([_hand(np.array([0.01 * t, 0, 0.5])) for t in range(10)])
    tilt = taskspace.euler_to_matrix([0.5, 0, 0])
    tilted = lm @ tilt
    ref = taskspace.to_trajectory(lm, 30, window=1)
    out = taskspace.to_trajectory(tilted, 30, camera_rpy=(0.5, 0, 0), window=1)
    np.testing.assert_allclose(out.poses[:, :3], ref.poses[:, :3], atol=1e-9)
    np.testing.assert_allclose(taskspace.euler_to_matrix(out.poses[:, 3:]),
                               taskspace.euler_to_matrix(ref.poses[:, 3:]),
                               atol=1e-6)


def test_missing_frames_and_gripper():
    lm = np.stack([_hand(np.zeros(3), pinch=p) for p in (0.08, 0.08, 0.01)])
    lm[1] = np.nan
    traj = taskspace.to_trajectory(lm, 30, window=1)
    assert not np.isnan(traj.poses).any()
    assert traj.gripper.tolist() == [False, False, True]


def test_gimbal_lock_roundtrip():
    rot = taskspace.euler_to_matrix([0.3, np.pi / 2, 0.2])
    np.testing.assert_allclose(
        taskspace.euler_to_matrix(taskspace.matrix_to_euler(rot)), rot,
        atol=1e-9)


def test_left_hand_matches_mirrored_right_hand():
    right = _hand(np.zeros(3))[None]
    left = right.copy()
    left[0, [taskspace.INDEX_MCP, taskspace.PINKY_MCP]] = left[
        0, [taskspace.PINKY_MCP, taskspace.INDEX_MCP]]
    np.testing.assert_allclose(taskspace.hand_frame(left, left=True),
                               taskspace.hand_frame(right), atol=1e-9)


def test_gripper_hysteresis():
    states = taskspace.gripper_states([0.05, 0.03, 0.05, 0.07], 0.04, 0.06)
    assert states.tolist() == [False, True, True, False]
