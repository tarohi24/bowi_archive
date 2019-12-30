"""
Aggregate results
"""
import argparse
from pathlib import Path
import json
from typing import Dict

import numpy as np


def mean(data: Dict[str, float]) -> float:
    return np.mean(list(data.values()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('resdir', type=Path)
    args = parser.parse_args()
    res_dir: Path = args.resdir

    # load res
    with open(res_dir / 'clfres.json') as fin:
        data: Dict[str, float] = json.load(fin)
    print(f'mean: {mean(data)}')
    return 0


if __name__ == '__main__':
    exit(main())
