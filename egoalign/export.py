"""Stage 5: write validated episodes for VLA fine-tuning."""

import pathlib

import numpy as np

from egoalign import taskspace


def save_episode(path, result, instruction):
    """Saves an accepted pipeline result as a compressed .npz episode.

    Keys: `actions` (T, 7) x, y, z, roll, pitch, yaw, gripper; `joints` (T, 7)
    Panda joint angles; `embeddings` (T, D) or (0,) if no encoder ran;
    `instruction`; `fps`; `fov_deg`; and `camera_rot` (3, 3) / `camera_pos`
    (3,) when the camera was registered.

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
    camera = {}
    if result.extrinsics is not None:
        camera = {'camera_rot': result.extrinsics.rot,
                  'camera_pos': result.extrinsics.pos}
    np.savez_compressed(
        path,
        actions=np.hstack([traj.poses, traj.gripper[:, None]]).astype(
            np.float32),
        joints=result.verdict.joints.astype(np.float32),
        embeddings=embeddings,
        instruction=np.array(instruction),
        fps=np.array(traj.fps),
        fov_deg=np.array(result.fov_deg),
        **camera)
    return path


def load_trajectory(path):
    """Reads an exported episode as (Trajectory, scene kwargs, instruction).

    Scene kwargs place the sim camera where the video's camera was, if known.
    """
    data = np.load(path)
    actions = data['actions']
    traj = taskspace.Trajectory(actions[:, :6].astype(float),
                                actions[:, 6] > 0.5, float(data['fps']))
    fov = float(data['fov_deg']) if 'fov_deg' in data else 65.0
    scene = {'hfov_deg': fov}
    if 'camera_rot' in data:
        scene.update(cam_pos=tuple(data['camera_pos']),
                     cam_rot=data['camera_rot'])
    return traj, scene, str(data['instruction'])


def save_rollout(path, rollout, instruction, fps):
    """Saves a rendered sim rollout (images, states, actions) as .npz."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=rollout.images, states=rollout.states,
                        actions=rollout.actions,
                        instruction=np.array(instruction), fps=np.array(fps))
    return path
