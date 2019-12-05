import re
from typing import List, Pattern, Set

from nltk.corpus import stopwords as nltk_sw
from nltk.tokenize import RegexpTokenizer


stopwords: Set[str] = set(nltk_sw.words('english'))
tokenizer: RegexpTokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
not_a_word_pat: Pattern = re.compile(r'^[^a-z0-9]*$')


def get_all_tokens(text: str) -> List[str]:
    """
    Preprocessing + tokenize
    """
    tokens: List[str] = tokenizer.tokenize(text.lower())
    # remove stopwords
    tokens: List[str] = [w for w in tokens if w not in stopwords]  # type: ignore
    tokens: List[str] = [w for w in tokens  # type: ignore
                         if not_a_word_pat.match(w) is None
                         and not w.isdigit()]
    tokens: List[str] = [w.replace('(', '').replace(')', '').replace('-', '')  # type: ignore
                         for w in tokens]
    return tokens
