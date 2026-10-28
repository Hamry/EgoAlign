"""Tests for Stage 1 landmark lifting."""

import numpy as np

from egoalign import extract


def test_lift_recovers_translation():
    rng = np.random.default_rng(0)
    world = rng.uniform(-0.05, 0.05, (21, 3))
    t = np.array([0.1, -0.05, 0.6])
    w, h, fov = 1280, 720, 65.0
    f = w / 2 / np.tan(np.radians(fov) / 2)
    cam = world + t
    image = np.column_stack([(f * cam[:, 0] / cam[:, 2] + w / 2) / w,
                             (f * cam[:, 1] / cam[:, 2] + h / 2) / h,
                             np.zeros(21)])
    np.testing.assert_allclose(extract.lift(image, world, w, h, fov), cam,
                               atol=1e-9)


def test_lift_rejects_hand_behind_camera():
    world = np.zeros((21, 3))
    image = np.full((21, 3), 0.5)
    assert np.isnan(extract.lift(image, world, 640, 480, 65.0)).all()
