from pathlib import Path
import os

from elasticsearch import Elasticsearch


project_root: Path = Path(os.environ['PROJECT_ROOT'])

data_dir: Path = project_root.joinpath('data')
models_dir: Path = project_root.joinpath('models')
results_dir: Path = project_root.joinpath('results')
param_dir: Path = project_root.joinpath('params')
cache_dir: Path = project_root.joinpath('cache')

# Elasticsearch
es: Elasticsearch = Elasticsearch(os.environ['ES_URL'])
