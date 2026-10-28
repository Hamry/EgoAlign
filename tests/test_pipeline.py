"""End-to-end pipeline, export and API tests with a stubbed hand tracker."""

import io

from fastapi import testclient
import numpy as np
import pytest

from egoalign import api
from egoalign import export
from egoalign import extract
from egoalign import pipeline
from egoalign import taskspace
from egoalign import validate

FRAMES = 60


def _hand(wrist):
    """Camera-frame hand pointing along camera +y (robot down)."""
    lm = np.tile(wrist, (21, 1)).astype(float)
    lm[taskspace.MIDDLE_MCP] += [0, 0.09, 0]
    lm[taskspace.INDEX_MCP] += [-0.03, 0.08, 0]
    lm[taskspace.PINKY_MCP] += [0.03, 0.08, 0]
    lm[taskspace.THUMB_TIP] += [-0.04, 0.1, 0]
    lm[taskspace.INDEX_TIP] += [0.04, 0.1, 0]
    return lm


class _FakeTracker:
    """Replays a reach-down-and-lift motion; frame 5 has no hand."""

    def __init__(self, fps, fov_deg):
        del fps, fov_deg
        self.left_votes = -1
        self._i = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def process(self, rgb):
        del rgb
        i, self._i = self._i, self._i + 1
        if i == 5:
            return np.full((21, 3), np.nan)
        depth = 0.15 * np.sin(np.pi * i / (FRAMES - 1))
        return _hand(np.array([0.0, depth, 0.5]))


@pytest.fixture(autouse=True)
def fake_video(monkeypatch):
    monkeypatch.setattr(extract, 'HandTracker', _FakeTracker)
    monkeypatch.setattr(extract, 'video_fps', lambda path: 30.0)
    monkeypatch.setattr(extract, 'read_video', lambda path: (
        np.zeros((48, 64, 3), np.uint8) for _ in range(FRAMES)))


def _encoder(frames):
    return np.ones((len(frames), 4), np.float32)


def test_process_and_export(tmp_path):
    result = pipeline.process('clip.mp4', encoder=_encoder)
    assert result.accepted and result.verdict.valid
    assert set(result.timings) == {'extraction', 'alignment', 'mapping',
                                   'validation'}
    out = export.save_episode(tmp_path / 'ep.npz', result, 'pick up the mug')
    data = np.load(out)
    assert data['actions'].shape == (FRAMES, 7)
    assert data['joints'].shape == (FRAMES, 7)
    assert data['embeddings'].shape == (FRAMES, 4)
    assert str(data['instruction']) == 'pick up the mug'


def test_export_refuses_rejected(tmp_path):
    result = pipeline.process('clip.mp4')
    result.accepted = False
    with pytest.raises(ValueError):
        export.save_episode(tmp_path / 'ep.npz', result, 'x')


def test_api_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(api, 'get_encoder', lambda: None)
    client = testclient.TestClient(api.app)
    resp = client.post('/episodes', data={'instruction': 'pick'},
                       files={'video': ('a.mp4', io.BytesIO(b'x'))})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['accepted'] and body['frames'] == FRAMES
    assert (tmp_path / body['episode']).exists()


def test_trims_to_valid_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, 'validate', lambda traj: validate.Verdict(
        False, 'unreachable', 20, np.zeros((20, 7))))
    result = pipeline.process('clip.mp4', encoder=_encoder)
    assert result.accepted and len(result.trajectory) == 20
    data = np.load(export.save_episode(tmp_path / 'ep.npz', result, 'x'))
    for key in ('actions', 'joints', 'embeddings'):
        assert len(data[key]) == 20


def test_api_rejects_oversized_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(api, 'MAX_UPLOAD_BYTES', 4)
    client = testclient.TestClient(api.app)
    resp = client.post('/episodes', data={'instruction': 'pick'},
                       files={'video': ('a.mp4:x', io.BytesIO(b'12345'))})
    assert resp.status_code == 413
