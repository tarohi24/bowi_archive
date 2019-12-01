from __future__ import annotations  # noqa
from dataclasses import dataclass
from operator import itemgetter
from typing import Dict, List, Tuple

from dataclasses_json import dataclass_json

from models.utils.utils import uniq


@dataclass
class RankItem:
    """
    both a prediction result or a ground truth
    recall, precision and ap considere self as a ground truth
    """
    query_id: str
    scores: Dict[Tuple[str, str], float]  # (docid, tag) -> score

    def get_doc_scores(self) -> Dict[str, float]:
        return {key[0]: score for key, score in self.scores.items()}

    def get_tag_scores(self) -> Dict[str, float]:
        return {key[1]: score for key, score in self.scores.items()}

    def pred_tags(self,
                  n_top: int) -> List[str]:
        sorted_score: List[str] = [
            tag for tag, _ in sorted(self.get_tag_scores().items(),
                                     key=itemgetter(1),
                                     reverse=True)]
        unique_tags: List[str] = uniq(orig=sorted_score, n_top=n_top)
        return unique_tags

    def __len__(self) -> int:
        return len(self.scores)


@dataclass_json
@dataclass
class ColDocument:
    docid: str
    title: str
    text: str
    tags: List[str]

    @classmethod
    def mapping(cls) -> Dict:
        return {
            'properties': {
                'docid': {
                    'type': 'keyword'
                },
                'title': {
                    'type': 'text',
                    'analyzer': 'english'
                },
                'text': {
                    'type': 'text',
                    'analyzer': 'english'
                },
                'tags': {
                    'type': 'keyword'
                }
            }
        }
