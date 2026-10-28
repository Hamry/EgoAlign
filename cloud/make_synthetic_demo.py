"""Writes a scripted pick as a registered demo .npz for a no-video dry run.

Usage: python cloud/make_synthetic_demo.py work/demo/synthetic.npz
Then:  egoalign augment work/demo/synthetic.npz ...
"""

import argparse
import pathlib

import numpy as np

from egoalign import sim

DOWN = (np.pi, 0.0, 0.0)


def _segment(a, b, frames):
    t = np.linspace(0, 1, frames, endpoint=False)[:, None]
    return np.r_[a] + (np.r_[b] - np.r_[a]) * t


def scripted_pick(mug_xy, fps=30):
    """(T, 7) xyz-rpy-gripper actions: reach, descend, close, lift."""
    grasp = (*mug_xy, 0.07, *DOWN)
    above = (*mug_xy, 0.2, *DOWN)
    reach = np.vstack([_segment((0.4, 0.0, 0.35, *DOWN), above, fps),
                       _segment(above, grasp, fps)])
    lift = np.vstack([_segment(grasp, (*mug_xy, 0.3, *DOWN), fps),
                      [(*mug_xy, 0.3, *DOWN)]])
    grip = np.r_[np.zeros(len(reach)), np.ones(len(lift))]
    return np.c_[np.vstack([reach, lift]), grip].astype(np.float32)


def main():
    """Writes the demo file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=pathlib.Path)
    parser.add_argument('--mug-xy', type=float, nargs=2, default=(0.5, 0.0))
    parser.add_argument('--instruction', default='pick up the mug')
    parser.add_argument('--cam-pos', type=float, nargs=3,
                        default=(1.0, 0.0, 0.6))
    args = parser.parse_args()
    target = (*args.mug_xy, sim.MUG_HALF_HEIGHT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, actions=scripted_pick(args.mug_xy), fps=np.array(30.0),
        fov_deg=np.array(65.0), instruction=np.array(args.instruction),
        camera_rot=sim.look_at(args.cam_pos, target),
        camera_pos=np.array(args.cam_pos))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
