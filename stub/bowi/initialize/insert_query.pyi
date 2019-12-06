from bowi import settings as settings
from pathlib import Path
from typing import Any, Dict, Generator

logger: Any

def load_query(dump_path: Path, es_index: str) -> Generator[Dict, None, None]: ...
def main() -> None: ...
