"""Stage 5: write validated episodes for VLA fine-tuning."""

import pathlib

import numpy as np


def save_episode(path, result, instruction):
    """Saves an accepted pipeline result as a compressed .npz episode.

    Keys: `actions` (T, 7) x, y, z, roll, pitch, yaw, gripper; `joints` (T, 7)
    Panda joint angles; `embeddings` (T, D) or (0,) if no encoder ran;
    `instruction`; `fps`.

    Args:
        path: Output file path.
        result: Accepted `pipeline.Result`.
        instruction: Natural-language task description.

    Returns:
        The written path.

    Raises:
        ValueError: If the result was rejected.
    """
    if not result.accepted:
        raise ValueError('refusing to export a rejected trajectory')
    traj = result.trajectory
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = (np.zeros(0, np.float32) if result.embeddings is None
                  else result.embeddings)
    np.savez_compressed(
        path,
        actions=np.hstack([traj.poses, traj.gripper[:, None]]).astype(
            np.float32),
        joints=result.verdict.joints.astype(np.float32),
        embeddings=embeddings,
        instruction=np.array(instruction),
        fps=np.array(traj.fps))
    return path
