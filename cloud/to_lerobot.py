"""Converts `egoalign augment` rollouts into a LeRobot dataset.

Run where `lerobot[smolvla,dataset]==0.6.1` is installed.
"""

import argparse
import pathlib

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

JOINTS = [f'joint{i}' for i in range(1, 8)]
# smolvla_base declares camera1..3; a subset passes its feature check.
IMAGE_KEY = 'observation.images.camera1'


def features(height, width):
    """LeRobot feature spec for POV image, 8-D state and 8-D action."""
    return {
        IMAGE_KEY: {'dtype': 'video', 'shape': (height, width, 3),
                    'names': ['height', 'width', 'channels']},
        'observation.state': {'dtype': 'float32', 'shape': (8,),
                              'names': JOINTS + ['gripper_width']},
        'action': {'dtype': 'float32', 'shape': (8,),
                   'names': JOINTS + ['gripper_close']},
    }


def main():
    """Writes every rollout in the directory as one LeRobot episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollouts', type=pathlib.Path)
    parser.add_argument('--repo-id', default='local/egoalign_pick')
    parser.add_argument('--root', type=pathlib.Path, required=True)
    args = parser.parse_args()

    files = sorted(args.rollouts.glob('*.npz'))
    if not files:
        raise SystemExit(f'no rollouts in {args.rollouts}')
    first = np.load(files[0])
    height, width = first['images'].shape[1:3]
    dataset = LeRobotDataset.create(
        args.repo_id, round(float(first['fps'])), features(height, width),
        root=args.root, robot_type='panda', use_videos=True)
    for path in files:
        ep = np.load(path)
        task = str(ep['instruction'])
        for image, state, action in zip(ep['images'], ep['states'],
                                        ep['actions']):
            dataset.add_frame({IMAGE_KEY: image, 'observation.state': state,
                               'action': action, 'task': task})
        dataset.save_episode()
    dataset.finalize()
    print(f'wrote {len(files)} episodes to {args.root}')


if __name__ == '__main__':
    main()
