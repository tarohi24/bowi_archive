import pytest

from pathlib import Path

from bowi.models import Document

from bowi.initialize.converters.cmu import ItemNotFound, CmuConveter


@pytest.fixture
def converter():
    return CmuConveter()


def test_get_itemize(converter):
    with pytest.raises(ItemNotFound):
        converter._get_itemize('')
    assert converter._get_itemize('key: value') == ('key', 'value')
    line: str = (
        'References: <1p9bseINNi6o@gap.caltech.edu>'
        '<1pamva$b6j@fido.asd.sgi.com> <1pcq4pINNqp1@gap.caltech.edu> <11702@vice.ICO.TE'
    )
    key, val = converter._get_itemize(line)
    assert key == 'References'
    assert val[:5] == '<1p9b'
    assert val[-5:] == 'CO.TE'


def test_get_document(tmpdir, converter):
    sample_dir: Path = Path(tmpdir) / 'rec.autos'
    sample_dir.mkdir()
    fpath: Path = sample_dir / '101557'
    with open(fpath, 'w') as fout:
        fout.write(
            '''
Newsgroups: rec.autos
Path: cantaloupe.srv.cs.cmu.edu!rochester!udel!gatech!howland.reston.ans.net!newsserver.jvnc.net!siemens!siemens.com!tjo
From: tjo@scr.siemens.com (Tom Ostrand)
Subject: Radio for Toyota Tercel
Message-ID: <tjo.734036386@siemens.com>
Keywords: radio,Tercel,replacement
Sender: news@scr.siemens.com (NeTnEwS)
Nntp-Posting-Host: bugatti.siemens.com
Organization: Siemens Corporate Research, Princeton (Plainsboro), NJ
Date: Mon, 5 Apr 1993 18:59:46 GMT
Lines: 19

I'm looking for a replacement radio/tape player for a 1984
Toyota Tercel.  Standard off-the-shelf unit is fine, but
every place I've gone to (Service Merchandise, etc.) doesn't
have my car in its model application book.  I want to just
take out the old radio, and slide in the new, with minimal time
spent hooking it up and adjusting the dashboard.

If you have put in a new unit in a similar car, I'd like to hear
what brand,  how easy it was to do the change, and any other
relevant information.

Please answer via E-mail.
Thanks,  Tom Ostrand

--
Tom Ostrand			E-mail:  tjo@scr.siemens.com
Siemens Corporate Research	Phone:   609-734-6569
755 College Road East		FAX:     609-734-6565
Princeton, NJ  08540-6668
''')
    doc: Document = converter.to_document(fpath)[0]
    assert doc.title == 'Radio for Toyota Tercel'
    assert doc.tags == ['rec.autos', ]
    assert doc.text[:5] == "I'm l"
    assert doc.text[-5:] == "-6668"
    assert doc.docid == '101557'
