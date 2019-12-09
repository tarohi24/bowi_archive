import re
from typing import Dict, List, Pattern, Set

from nltk.corpus import stopwords as nltk_sw

from bowi.settings import es


stopwords: Set[str] = set(nltk_sw.words('english'))
not_a_word_pat: Pattern = re.compile(r'^[^a-z0-9]*$')


def get_all_tokens(text: str) -> List[str]:
    """
    Preprocessing + tokenize
    """
    tokens: List[str] = stem_words(text)
    # While stem_words removes some stopwords, nltk stopword list is richer
    tokens: List[str] = [w for w in tokens if w not in stopwords]  # type: ignore
    tokens: List[str] = [w for w in tokens  # type: ignore
                         if not_a_word_pat.match(w) is None and not w.isdigit()]
    tokens: List[str] = [w.replace('(', '').replace(')', '').replace('-', '')  # type: ignore
                         for w in tokens]
    return tokens


def stem_words(text: str) -> List[str]:
    if len(text) == 0:
        return []
    head, tail = text[:9000], text[9000:]
    body: Dict = {"analyzer": "english", "text": head}
    res: Dict = es.indices.analyze(body=body)
    tokens: List[str] = [tok['token'] for tok in res['tokens']]
    return tokens + stem_words(tail)
