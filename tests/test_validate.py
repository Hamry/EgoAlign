"""KPI 3: the kinematic filter rejects every adversarial trajectory."""

import numpy as np
import pytest

from egoalign import taskspace
from egoalign import validate

DOWN = (np.pi, 0.0, 0.0)
ANGLED = (np.pi, -np.pi / 4, 0.0)


def _path(*waypoints, frames=20):
    """Linearly interpolates (xyz, rpy) waypoints into a 30 fps trajectory."""
    poses = [np.r_[a] + (np.r_[b] - np.r_[a]) * t
             for (a, b) in zip(waypoints, waypoints[1:])
             for t in np.linspace(0, 1, frames, endpoint=False)]
    poses.append(np.r_[waypoints[-1]])
    poses = np.array(poses)
    return taskspace.Trajectory(poses, np.zeros(len(poses), bool), 30.0)


VALID = {
    'top_down_pick_place': _path((0.5, 0.1, 0.25, *DOWN),
                                 (0.5, 0.1, 0.06, *DOWN),
                                 (0.5, 0.1, 0.25, *DOWN),
                                 (0.4, -0.2, 0.25, *DOWN)),
    'angled_approach': _path((0.5, 0.0, 0.3, *ANGLED),
                             (0.6, 0.0, 0.15, *ANGLED)),
}

ADVERSARIAL = {
    'through_table': ('table_collision', _path(
        (0.5, 0.0, 0.2, *DOWN), (0.5, 0.0, -0.08, *DOWN))),
    # Joints whip near the stretched-out singularity before IK fails.
    'out_of_reach': ('joint_velocity', _path(
        (0.5, 0.0, 0.3, *DOWN), (1.2, 0.0, 0.3, *DOWN), frames=120)),
    'through_base': ('unreachable', _path(
        (0.4, 0.0, 0.3, *DOWN), (0.0, 0.0, 0.3, *DOWN), frames=120)),
    'wrist_spin': ('joint_velocity', _path(
        (0.5, 0.0, 0.3, np.pi, 0.0, 0.0),
        (0.5, 0.0, 0.3, np.pi, 0.0, 2 * np.pi), frames=10)),
    'teleport': ('joint_velocity', _path(
        (0.5, 0.2, 0.3, *DOWN), (0.5, -0.2, 0.3, *DOWN), frames=1)),
}


@pytest.mark.parametrize('name', VALID)
def test_feasible_trajectory_passes(name):
    verdict = validate.validate(VALID[name])
    assert verdict.valid, (verdict.reason, verdict.frame)
    assert len(verdict.joints) == len(VALID[name])


@pytest.mark.parametrize('name', ADVERSARIAL)
def test_adversarial_trajectory_rejected(name):
    reason, traj = ADVERSARIAL[name]
    verdict = validate.validate(traj)
    assert not verdict.valid
    assert verdict.reason == reason


def test_empty_trajectory_rejected():
    empty = taskspace.Trajectory(np.zeros((0, 6)), np.zeros(0, bool), 30.0)
    assert validate.validate(empty).reason == 'empty'
