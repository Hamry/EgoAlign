"""Scene tests; first run downloads the Menagerie Panda (cached)."""

import mujoco
import numpy as np

from egoalign import robot
from egoalign import sim


def test_scene_ik_and_render():
    model = sim.build(mug_xy=(0.5, 0.1))
    panda = robot.Panda(model)
    q, ok = panda.solve(np.r_[0.5, 0.1, 0.2, np.pi, 0, 0], robot.READY)
    assert ok
    np.testing.assert_allclose(panda.data.site('tcp').xpos, [0.5, 0.1, 0.2],
                               atol=0.01)
    assert abs(sim.mug_height(panda.data)) < 1e-6
    with mujoco.Renderer(model, 240, 320) as renderer:
        renderer.update_scene(panda.data, 'pov')
        assert renderer.render().shape == (240, 320, 3)


def test_look_at_points_camera_at_target():
    rot = sim.look_at([1, 0, 0.5], [0, 0, 0.5])
    np.testing.assert_allclose(rot[:, 2], [-1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(rot[:, 1], [0, 0, -1], atol=1e-9)


def test_look_at_straight_down_is_finite():
    rot = sim.look_at([0.5, 0, 1], [0.5, 0, 0])
    assert np.isfinite(rot).all()
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-9)


def test_sim_model_ready_pose_has_no_arm_collisions():
    panda = robot.Panda(sim.build())
    panda.forward(robot.READY)
    assert panda.collisions() == []
