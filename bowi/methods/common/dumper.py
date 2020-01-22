from pathlib import Path
import datetime
from typing import List

from bowi.methods.common.types import Context, TRECResult
from bowi.models import Document
from bowi.settings import results_dir


def get_dump_dir(context: Context) -> Path:
    path: Path = results_dir.joinpath(
        f"{context.es_index}/{context.method}/{context.runname}")
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        pass
    return path


def dump_prel(res: TRECResult,
              context: Context) -> None:
    path: Path = get_dump_dir(context=context).joinpath('pred.prel')
    with open(path, 'a') as fout:
        fout.write(res.to_prel())
        fout.write('\n')


def dump_time(start_time: datetime.datetime,
              context: Context) -> None:
    path: Path = get_dump_dir(context=context) / 'time.txt'
    with open(path, 'a') as fout:
        fout.write(str(start_time))
        fout.write('\n')
