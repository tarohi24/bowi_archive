from bowi.utils.utils import uniq as uniq
from typing import Any, Dict, List, Tuple

class RankItem:
    query_id: str
    scores: Dict[Tuple[str, str], float]
    def get_doc_scores(self) -> Dict[str, float]: ...
    def get_tag_scores(self) -> Dict[str, float]: ...
    def pred_tags(self, n_top: int) -> List[str]: ...
    def __len__(self) -> int: ...
    def __init__(self, query_id: Any, scores: Any) -> None: ...

class Document:
    docid: str
    title: str
    text: str
    tags: List[str]
    @classmethod
    def mapping(cls: Any) -> Dict: ...
    def __init__(self, docid: Any, title: Any, text: Any, tags: Any) -> None: ...