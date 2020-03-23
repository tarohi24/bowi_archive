from bowi.methods.common.loader import load_query_files
from bowi.elas.client import EsClient
from collections import defaultdict
from tqdm import tqdm
from bowi.embedding.base import mat_normalize
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

for dataset in ['clef', ]:

    rels = defaultdict(list)
    with open(f'notebooks/gt/{dataset}.qrel') as fin:
        for line in fin.readlines():
            qid, _, relid, _ = line.split()
            rels[qid].append(relid)

    q_escl = EsClient(f'{dataset}_query')
    escl = EsClient(dataset)
    data = dict()
    for qid, relid_list in tqdm(rels.items()):
        try:
            q_tokens = q_escl.get_tokens_from_doc(qid)
        except:
            continue
        col_tokens = []
        for rid in relid_list:
            try:
                tokens = escl.get_tokens_from_doc(rid)
            except:
                tokens = ['']
            col_tokens.append(tokens)
                
        cvec = CountVectorizer()
        mat = cvec.fit_transform([' '.join(q_tokens)] + [' '.join(toks) for toks in col_tokens]).toarray()
        vec, x = mat[0], mat[1:]
        x = mat_normalize(x)
        data[qid] = {rid: sim for rid, sim in zip(relid_list, np.dot(x, vec) / np.linalg.norm(vec))}
    
    with open(f'notebooks/bow_sim/{dataset}.py', 'w') as fout:
        json.dump(data, fout)
