"""Command line: `egoalign process VIDEO` and `egoalign serve`."""

import argparse
import json
import math
import pathlib

from egoalign import align
from egoalign import augment
from egoalign import export
from egoalign import pipeline
from egoalign import register


def _process(args):
    encoder = None if args.no_embed else align.Encoder()
    extrinsics = None
    if args.marker_xy is not None:
        extrinsics = register.register_video(args.video, args.fov,
                                             args.marker_xy, args.marker_size)
    result = pipeline.process(
        args.video, encoder,
        camera_rpy=tuple(math.radians(d) for d in args.camera_rpy),
        fov_deg=args.fov, extrinsics=extrinsics)
    report = result.summary()
    if result.accepted:
        out = args.out / f'{pathlib.Path(args.video).stem}.npz'
        report['episode'] = str(export.save_episode(out, result,
                                                    args.instruction))
    print(json.dumps(report, indent=2))


def _augment(args):
    traj, scene, instruction = export.load_trajectory(args.episode)
    if 'cam_rot' not in scene:
        raise SystemExit('episode has no camera registration; re-run '
                         '`egoalign process` with --marker-xy')
    stem = pathlib.Path(args.episode).stem
    episodes = augment.generate(traj, args.episodes, scene_kwargs=scene,
                                radius=args.radius, seed=args.seed,
                                width=args.size, height=args.size)
    count = 0
    for count, (mug_xy, rollout) in enumerate(episodes, 1):
        out = export.save_rollout(args.out / f'{stem}_{count:03d}.npz',
                                  rollout, instruction, traj.fps)
        print(f'{out}  mug at {mug_xy.round(3).tolist()}')
    print(f'{count}/{args.episodes} successful episodes')


def _serve(args):
    import uvicorn  # pylint: disable=import-outside-toplevel
    uvicorn.run('egoalign.api:app', host=args.host, port=args.port)


def main(argv=None):
    """Parses arguments and runs the chosen subcommand."""
    parser = argparse.ArgumentParser(prog='egoalign')
    sub = parser.add_subparsers(required=True)

    proc = sub.add_parser('process', help='turn one video into an episode')
    proc.add_argument('video')
    proc.add_argument('--instruction', required=True)
    proc.add_argument('--out', type=pathlib.Path, default='egoalign_data')
    proc.add_argument('--camera-rpy', type=float, nargs=3, default=(0, 0, 0),
                      metavar=('ROLL', 'PITCH', 'YAW'),
                      help='camera tilt from canonical view [deg]')
    proc.add_argument('--fov', type=float, default=65.0,
                      help='horizontal field of view [deg]')
    proc.add_argument('--marker-xy', type=float, nargs=2, metavar=('X', 'Y'),
                      help='ArUco marker centre in robot frame [m]; enables '
                      'absolute placement')
    proc.add_argument('--marker-size', type=float, default=0.1,
                      help='printed marker side length [m]')
    proc.add_argument('--no-embed', action='store_true',
                      help='skip CLIP embeddings')
    proc.set_defaults(func=_process)

    aug = sub.add_parser('augment',
                         help='render sim robot episodes from a demo')
    aug.add_argument('episode', help='.npz from `process` with --marker-xy')
    aug.add_argument('--episodes', type=int, default=50)
    aug.add_argument('--radius', type=float, default=0.08,
                     help='max mug offset from the demo [m]')
    aug.add_argument('--size', type=int, default=256, help='image size [px]')
    aug.add_argument('--seed', type=int, default=0)
    aug.add_argument('--out', type=pathlib.Path, default='rollouts')
    aug.set_defaults(func=_augment)

    serve = sub.add_parser('serve', help='run the upload API')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8000)
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    if getattr(args, 'marker_size', 1) <= 0:
        parser.error('--marker-size must be positive')
    args.func(args)


if __name__ == '__main__':
    main()
