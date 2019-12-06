from nltk.tokenize import RegexpTokenizer
from typing import List, Pattern, Set

stopwords: Set[str]
tokenizer: RegexpTokenizer
not_a_word_pat: Pattern

def get_all_tokens(text: str) -> List[str]: ...
