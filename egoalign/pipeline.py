"""Runs the extraction -> alignment -> mapping -> validation DAG on a video."""

import contextlib
import dataclasses
import time

import numpy as np

from egoalign import align
from egoalign import extract
from egoalign import taskspace
from egoalign import validate


@dataclasses.dataclass
class Result:
    """Pipeline output for one video.

    Attributes:
        trajectory: Robot-frame trajectory, trimmed to the valid prefix.
        embeddings: (T, D) frame embeddings aligned to `trajectory`, or None.
        verdict: Kinematic validation verdict for the full trajectory.
        accepted: True if the (possibly trimmed) trajectory is exportable.
        frames: Number of decoded video frames.
        timings: Seconds spent per stage.
    """

    trajectory: taskspace.Trajectory
    embeddings: np.ndarray | None
    verdict: validate.Verdict
    accepted: bool
    frames: int
    timings: dict

    @property
    def throughput_fps(self):
        """Video frames processed per second of pipeline time."""
        return self.frames / max(sum(self.timings.values()), 1e-9)

    def summary(self):
        """JSON-friendly report of the outcome."""
        return {
            'accepted': self.accepted,
            'valid': self.verdict.valid,
            'reason': self.verdict.reason,
            'failed_frame': self.verdict.frame,
            'frames': self.frames,
            'kept_frames': len(self.trajectory) if self.accepted else 0,
            'throughput_fps': self.throughput_fps,
            'timings': self.timings,
        }


def _extract(video_path, fps, fov_deg, keep_thumbnails, max_frames):
    """Returns (landmarks, thumbnails, is_left) from one pass over the video.

    Raises:
        ValueError: If the video has more than `max_frames` frames.
    """
    landmarks, thumbs = [], []
    with (extract.HandTracker(fps, fov_deg) as tracker,
          contextlib.closing(extract.read_video(video_path)) as frames):
        for frame in frames:
            if len(landmarks) == max_frames:
                raise ValueError(f'video longer than {max_frames} frames')
            landmarks.append(tracker.process(frame))
            if keep_thumbnails:
                thumbs.append(align.thumbnail(frame))
    left = tracker.left_votes > 0
    return np.array(landmarks).reshape(-1, 21, 3), thumbs, left


def process(video_path, encoder=None, camera_rpy=(0.0, 0.0, 0.0),
            fov_deg=65.0, trim=True, min_frames=15, max_frames=3600):
    """Turns a POV video into a validated robot trajectory.

    Args:
        video_path: Input video file.
        encoder: Optional `align.Encoder`; skips embeddings when None.
        camera_rpy: Camera tilt from the canonical view [rad].
        fov_deg: Horizontal camera field of view [deg].
        trim: Keep the valid prefix of an invalid trajectory.
        min_frames: Shortest prefix worth keeping when trimming.
        max_frames: Longest video accepted, bounding memory.

    Returns:
        A `Result`.

    Raises:
        ValueError: If the video can't be read, is too long or shows no hand.
    """
    timings = {}
    t0 = time.perf_counter()
    fps = extract.video_fps(video_path)
    landmarks, thumbs, left = _extract(video_path, fps, fov_deg,
                                       encoder is not None, max_frames)
    timings['extraction'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    traj = taskspace.to_trajectory(landmarks, fps, camera_rpy, left=left)
    timings['mapping'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    verdict = validate.validate(traj)
    timings['validation'] = time.perf_counter() - t0

    accepted = verdict.valid
    if not verdict.valid and trim and verdict.frame >= min_frames:
        traj = traj[:verdict.frame]
        accepted = True

    # Embed only exported frames; the encoder is the costliest stage.
    t0 = time.perf_counter()
    embeddings = None
    if encoder is not None and accepted:
        embeddings = encoder(thumbs[:len(traj)])
    timings['alignment'] = time.perf_counter() - t0
    return Result(traj, embeddings, verdict, accepted, len(landmarks), timings)
