import re
from typing import Pattern, Set

from nltk.corpus import stopwords as nltk_sw
from nltk.tokenize import RegexpTokenizer


stopwords: Set[str] = set(nltk_sw.words('english'))
tokenizer: RegexpTokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
not_a_word_pat: Pattern = re.compile(r'^[^a-z0-9]*$')


def is_valid_word(word: str) -> bool:
    if not_a_word_pat.match(word) is not None:
        return False
    elif word.isdigit():
        return False
    else:
        return True
