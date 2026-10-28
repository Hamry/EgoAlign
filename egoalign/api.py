"""FastAPI ingestion endpoint: upload a POV video, get a validated episode."""

import functools
import logging
import math
import os
import pathlib
import re
import tempfile
import threading
import uuid

import fastapi

from egoalign import align
from egoalign import export
from egoalign import pipeline

OUTPUT_DIR = pathlib.Path(os.environ.get('EGOALIGN_OUTPUT', 'egoalign_data'))
MAX_UPLOAD_BYTES = int(os.environ.get('EGOALIGN_MAX_UPLOAD_MB', '500')) << 20
_CHUNK = 1 << 20
_encoder_lock = threading.Lock()

app = fastapi.FastAPI(title='EgoAlign')


def get_encoder():
    """Returns the shared encoder, or None if it can't be loaded."""
    with _encoder_lock:
        return _load_encoder()


@functools.cache
def _load_encoder():
    try:
        return align.Encoder()
    except (ImportError, OSError) as e:
        logging.warning('alignment disabled, encoder unavailable: %s', e)
        return None


def _save_upload(upload, dst):
    """Copies an upload to `dst`, enforcing MAX_UPLOAD_BYTES."""
    size = 0
    while chunk := upload.file.read(_CHUNK):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise fastapi.HTTPException(413, 'upload too large')
        dst.write(chunk)


def _safe_suffix(filename):
    """Returns the filename's extension if it is plain alphanumeric."""
    suffix = pathlib.Path(filename or '').suffix
    return suffix if re.fullmatch(r'\.[A-Za-z0-9]{1,5}', suffix) else ''


@app.post('/episodes')
def create_episode(video: fastapi.UploadFile,
                   instruction: str = fastapi.Form(..., min_length=1,
                                                   max_length=500),
                   camera_roll_deg: float = fastapi.Form(0.0, ge=-180, le=180),
                   camera_pitch_deg: float = fastapi.Form(0.0, ge=-180, le=180),
                   camera_yaw_deg: float = fastapi.Form(0.0, ge=-180, le=180),
                   fov_deg: float = fastapi.Form(65.0, gt=10, lt=170)):
    """Processes an uploaded video and stores it if it passes validation."""
    with tempfile.NamedTemporaryFile(suffix=_safe_suffix(video.filename),
                                     delete=False) as f:
        pass
    try:
        with open(f.name, 'wb') as dst:
            _save_upload(video, dst)
        result = pipeline.process(
            f.name, encoder=get_encoder(),
            camera_rpy=tuple(math.radians(d) for d in (
                camera_roll_deg, camera_pitch_deg, camera_yaw_deg)),
            fov_deg=fov_deg)
    except ValueError as e:
        raise fastapi.HTTPException(422, str(e)) from e
    finally:
        os.unlink(f.name)
    episode = None
    if result.accepted:
        episode = export.save_episode(
            OUTPUT_DIR / f'{uuid.uuid4().hex}.npz', result, instruction).name
    return {**result.summary(), 'episode': episode}
