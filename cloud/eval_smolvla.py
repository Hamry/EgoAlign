"""KPI 4: closed-loop success rate of a fine-tuned SmolVLA in the sim scene.

Run where `lerobot[smolvla]==0.6.1` and egoalign are installed.
"""

import argparse

import cv2
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import mujoco
import numpy as np
import torch

from egoalign import augment
from egoalign import export
from egoalign import replay
from egoalign import sim

# smolvla_base declares camera1..3; a subset passes its feature check.
IMAGE_KEY = 'observation.images.camera1'


def make_actor(policy, pre, post, device, task):
    """Returns f(image, state) -> 8-D numpy action using the full pipeline."""

    def act(image, state):
        pixels = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255
        batch = {IMAGE_KEY: pixels.to(device),
                 'observation.state': torch.from_numpy(state)[None].to(device),
                 'task': [task]}
        with torch.inference_mode():
            action = post(policy.select_action(pre(batch)))
        return action[0].float().cpu().numpy()

    return act


def run_episode(act, model, q0, fps, max_steps, size, frames=None):
    """Runs the actor until the mug is lifted or time runs out."""
    substeps = replay.control_substeps(model, fps)
    data = replay.reset(model, q0)
    with mujoco.Renderer(model, size, size) as renderer:
        for _ in range(max_steps):
            image, state = replay.observe(renderer, data)
            if frames is not None:
                frames.append(image)
            replay.apply_action(data, act(image, state))
            mujoco.mj_step(model, data, substeps)
            if sim.mug_height(data) >= replay.LIFT_SUCCESS:
                return True
    return False


def write_video(path, frames, fps):
    """Writes RGB frames to an .mp4.

    Raises:
        RuntimeError: If no mp4 encoder is available.
    """
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'cannot open video writer for {path}')
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    """Evaluates a checkpoint over randomized mug placements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True,
                        help='fine-tuned checkpoint (pretrained_model dir)')
    parser.add_argument('--demo', required=True,
                        help='registered demo .npz from `egoalign process`')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--radius', type=float, default=0.08)
    parser.add_argument('--max-steps', type=int,
                        help='default: twice the demo length')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--video', help='optional .mp4 of the first episode')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be at least 1')

    traj, scene, task = export.load_trajectory(args.demo)
    max_steps = args.max_steps or 2 * len(traj)
    src = traj.poses[augment.grasp_frame(traj), :2]
    start = replay.joint_path(sim.build(mug_xy=tuple(src), **scene), traj[:1])
    if start is None:
        raise SystemExit('demo start pose is unreachable')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = SmolVLAPolicy.from_pretrained(args.policy).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.policy,
        preprocessor_overrides={'device_processor': {'device': device}})
    act = make_actor(policy, pre, post, device, task)

    rng = np.random.default_rng(args.seed)
    successes = 0
    for i in range(args.episodes):
        mug = src + rng.uniform(-args.radius, args.radius, 2)
        model = sim.build(mug_xy=tuple(mug), width=args.size,
                          height=args.size, **scene)
        policy.reset()
        frames = [] if args.video and i == 0 else None
        ok = run_episode(act, model, start[0], traj.fps, max_steps, args.size,
                         frames)
        successes += ok
        print(f'episode {i}: mug {mug.round(3).tolist()} '
              f'{"success" if ok else "fail"}')
        if frames:
            write_video(args.video, frames, traj.fps)
    print(f'success rate: {successes}/{args.episodes} '
          f'= {successes / args.episodes:.0%} (KPI 4 target >= 70%)')


if __name__ == '__main__':
    main()
