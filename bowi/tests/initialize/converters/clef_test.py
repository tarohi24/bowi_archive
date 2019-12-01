from pathlib import Path
import unittest
from typing import List
import xml.etree.ElementTree as ET

import pytest

from bowi.initialize.converters.clef import CLEFConverter, NoneException
from bowi.settings import data_dir


class CLEFConverterTest(unittest.TestCase):

    def to_xml_root(self, docid: str) -> ET.Element:
        """
        Parameters
        ----------
        docid
            EP... *with* '-'
            e.g. EP-0050001-A2
        """
        first: str = docid[3]
        second: str = docid[4:6]
        third: str = docid[6:8]
        forth: str = docid[8:10]

        fpath: Path = data_dir.joinpath(
            f'clef/orig/collection/00000{first}/{second}/{third}/{forth}/{docid}.xml')

        root: ET.Element = ET.parse(str(fpath.resolve())).getroot()
        return root

    def __init__(self, *args, **kwargs):
        super(CLEFConverterTest, self).__init__(*args, **kwargs)
        self.converter: CLEFConverter = CLEFConverter()
        self.docids: List[str] = [
            'EP-0050001-A2',
            'EP-1010180-B1'
        ]
        self.roots: List[ET.Element] = [self.to_xml_root(docid) for docid in self.docids]

    def test_get_title(self):
        titles: List[str] = [
            'A golf aid.',
            'A READ-ONLY MEMORY AND READ-ONLY MEMORY DEVICE',
        ]
        self.assertListEqual(
            titles,
            [self.converter._get_title(root) for root in self.roots])

    def test_get_docid(self):
        self.assertListEqual(
            [docid.replace('-', '') for docid in self.docids],
            [self.converter._get_docid(root) for root in self.roots])

    def test_get_tags(self):
        self.assertListEqual(
            [set('A63B'.split()),
             set('H01L G11C'.split())],
            [set(self.converter._get_tags(root)) for root in self.roots])

    def test_get_text(self):
        with pytest.raises(NoneException):
            self.converter._get_text(self.roots[0])

        self.assertListEqual(
            'The present invention'.split(),
            self.converter._get_text(self.roots[1]).split()[:3])

    def test_get_paras(self):
        with pytest.raises(NoneException):
            self.converter._get_paragraph_list(self.roots[0])

        assert len(self.converter._get_paragraph_list(self.roots[1])) == 40
