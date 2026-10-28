"""Augmentation: retargeting math and end-to-end generation."""

import numpy as np

from egoalign import pipeline
from egoalign import register
from egoalign import validate

from egoalign import augment
from egoalign import export
from egoalign import taskspace

import test_replay


def test_retarget_keeps_start_and_shifts_after_grasp():
    traj = test_replay._pick(close=True)  # pylint: disable=protected-access
    g = augment.grasp_frame(traj)
    out = augment.retarget(traj, (0.05, -0.02))
    np.testing.assert_allclose(out.poses[0], traj.poses[0])
    np.testing.assert_allclose(out.poses[g:, :2] - traj.poses[g:, :2],
                               np.tile([0.05, -0.02], (len(traj) - g, 1)))
    np.testing.assert_allclose(out.poses[:, 2:], traj.poses[:, 2:])


def test_generate_and_roundtrip(tmp_path):
    traj = test_replay._pick(close=True)  # pylint: disable=protected-access
    episodes = list(augment.generate(traj, 2, radius=0.03, width=64,
                                     height=64))
    assert len(episodes) == 2
    mug_xy, rollout = episodes[1]
    assert rollout.success and not np.allclose(mug_xy, test_replay.MUG)
    path = export.save_rollout(tmp_path / 'r.npz', rollout, 'pick', 30.0)
    assert np.load(path)['images'].shape == rollout.images.shape


def test_load_trajectory_roundtrip(tmp_path):
    poses = np.zeros((3, 6))
    np.savez(tmp_path / 'ep.npz',
             actions=np.c_[poses, [0, 1, 1]].astype(np.float32),
             fps=np.array(30.0), fov_deg=np.array(60.0),
             instruction=np.array('pick'), camera_rot=np.eye(3),
             camera_pos=np.ones(3))
    traj, scene, instruction = export.load_trajectory(tmp_path / 'ep.npz')
    assert isinstance(traj, taskspace.Trajectory)
    assert traj.gripper.tolist() == [False, True, True]
    assert scene['hfov_deg'] == 60.0 and scene['cam_pos'] == (1.0, 1.0, 1.0)
    assert instruction == 'pick'


def test_save_episode_feeds_load_trajectory(tmp_path):
    traj = test_replay._pick(close=True)  # pylint: disable=protected-access
    ext = register.Extrinsics(np.eye(3), np.array([1.0, 0.0, 0.5]))
    verdict = validate.Verdict(True, None, None, np.zeros((len(traj), 7)))
    result = pipeline.Result(traj, None, verdict, True, len(traj), {}, ext,
                             60.0)
    path = export.save_episode(tmp_path / 'ep.npz', result, 'pick')
    loaded, scene, _ = export.load_trajectory(path)
    np.testing.assert_allclose(loaded.poses, traj.poses, atol=1e-6)
    np.testing.assert_allclose(scene['cam_rot'], np.eye(3))
    assert scene['hfov_deg'] == 60.0
