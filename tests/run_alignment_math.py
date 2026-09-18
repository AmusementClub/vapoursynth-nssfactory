#!/usr/bin/env python3
"""CTest adapter for the independent NumPy/SciPy alignment oracle.

The oracle writes a fresh run directory so repeated CTest invocations retain
their inputs, outputs and summary instead of colliding with an earlier run.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as exc:
        print(f'alignment math oracle SKIP: {exc}', file=sys.stderr)
        return 77

    script = Path(__file__).with_name('test_alignment_math.py').resolve()
    base = Path(args.out).resolve()
    # Suite creates the output directory itself with exist_ok=False.
    run_dir = base / f'run-{os.getpid()}'
    base.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(script), '--probe', str(Path(args.probe).resolve()), '--out', str(run_dir)]
    runpy.run_path(str(script), run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
