import re


pat = re.compile(r'^(?P<qid>\w+)\s+Q0\s+(?P<relid>\w+)\s+(?P<rank>\d+)\s+(?P<score>.+)\sSTANDARD$')
with open('pred.prel') as fin:
    lines = fin.readlines()
docs = set()
qid_prev = ''
qids = []
invalids = set()
for i, line in enumerate(lines):
    match = pat.match(line)
    qid = match.group('qid')
    if qid != qid_prev:
        docs = set()
        qids.append(qid)
    relid = match.group('relid')
    if relid in docs:
        invalids.add(i)
    else:
        docs.add(relid)
    qid_prev = qid



with open('p.prel', 'w') as fout:
    for i, line in enumerate(lines):
        if i not in invalids:
            fout.write(line)
