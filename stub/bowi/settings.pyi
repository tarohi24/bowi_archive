from elasticsearch import Elasticsearch
from pathlib import Path

project_root: Path
data_dir: Path
models_dir: Path
results_dir: Path
param_dir: Path
cache_dir: Path
es: Elasticsearch
