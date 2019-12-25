import argparse
import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Type, TypeVar

import yaml
from typedflow.flow import Flow

from bowi.methods.common.types import Context, Param
from bowi.methods.common.methods import Method
from bowi.methods.common.dumper import get_dump_dir

# methods
from bowi.methods.methods import keywords, bm25
from bowi.methods.methods.fuzzy import naive, rerank
from bowi.initialize.cacher import embedding, pre_filtering, col_embs


M = TypeVar('M', bound=Method)


def get_method(method_name: str) -> Type[M]:
    if method_name == 'keywords':
        return keywords.KeywordBaseline
    elif method_name == 'fuzzy.naive':
        return naive.FuzzyNaive
    elif method_name == 'fuzzy.rerank':
        return rerank.FuzzyRerank
    elif method_name == 'cache.pre_filtering':
        return pre_filtering.PreSearcher
    elif method_name == 'cache.embedding':
        return embedding.EmbeddingCacher
    elif method_name == 'cache.colembs':
        return col_embs.ColEmbs
    elif method_name == 'bm25i':
        return bm25.BM25I
    else:
        raise KeyError(f'{method_name} is not found')


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('paramfile',
                        metavar='F',
                        type=Path,
                        nargs=1,
                        help='A yaml file')
    parser.add_argument('-d',
                        '--debug',
                        dest='debug',
                        default=False,
                        action='store_true')
    return parser


def parse(args: NamedTuple) -> List[M]:
    if args.debug:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)

    path: Path = args.paramfile[0]
    with open(path) as fin:
        data: Dict = yaml.load(fin, Loader=yaml.Loader)
    n_docs: int = data['n_docs']
    es_index: str = data['es_index']
    method_name: str = data['method']
    method_type: Type[Method] = get_method(method_name)
    param_type: Type = method_type.param_type

    lst: List[Method] = []
    for p in data['params']:
        runname: str = str(p['name'])
        context: Context = Context(
            n_docs=n_docs,
            es_index=es_index,
            method=method_name,
            runname=runname
        )
        del p['name']
        param: Param = param_type(**p)
        method: M = method_type(context=context, param=param)
        lst.append(method)
    return lst


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    methods: List[Method] = parse(args)
    for met in methods:
        dump_dir: Path = get_dump_dir(met.context)
        try:
            for path in dump_dir.glob('*'):
                path.unlink()
        except FileNotFoundError:
            pass
        flow: Flow = met.create_flow(debug=args.debug)
        flow.typecheck()
        flow.run()
    return 0


if __name__ == '__main__':
    exit(main())
