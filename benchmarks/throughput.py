"""KPI 1: end-to-end pipeline throughput on a real video (encoder warmed)."""

import argparse
import json

import numpy as np

from egoalign import align
from egoalign import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--no-embed', action='store_true')
    args = parser.parse_args()
    encoder = None if args.no_embed else align.Encoder()
    if encoder is not None:
        encoder([np.zeros((224, 224, 3), np.uint8)])
    result = pipeline.process(args.video, encoder)
    report = result.summary()
    counts = {k: result.frames for k in result.timings}
    if result.embeddings is None:
        counts['alignment'] = 0
    else:
        counts['alignment'] = report['kept_frames']
    report['stage_fps'] = {k: counts[k] / v for k, v in result.timings.items()
                           if counts[k] and v > 0}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
