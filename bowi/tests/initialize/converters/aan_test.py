from pathlib import Path
from more_itertools import flatten
import unittest
from typing import List

from bowi.initialize.converters.aan import AANConverter
from bowi.models import Document
from bowi.settings import data_dir


class AANConverterTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(AANConverterTest, self).__init__(*args, **kwargs)
        self.converter: AANConverter = AANConverter()
        self.source_dir: Path = data_dir.joinpath('aan/orig/collection')
        self.docs: List[Document] = list(flatten(
            [self.converter.to_document(
                self.source_dir.joinpath(f'{docid}.txt'))
             for docid in ['D07-1026', 'D07-1016', ]]))

    def test_get_title(self):
        assert self.docs[0].title.value == 'Instance Based Lexical Entailment for Ontology Population'

    def test_get_docid(self):
        assert self.docs[0].docid.value == 'D07-1026'

    def test_get_tags(self):
        self.assertListEqual(
            self.docs[0].tags.value,
            ['D07'])

    def test_get_text(self):
        assert self.docs[0].text.value.split()[:3] == 'Proceedings of the'.split()

    def test_get_paras(self):
        pass
