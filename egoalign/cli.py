"""Command line: `egoalign process VIDEO` and `egoalign serve`."""

import argparse
import json
import math
import pathlib

from egoalign import align
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
