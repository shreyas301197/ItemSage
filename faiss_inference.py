# !/Users/shreyasve/anaconda3/envs/env_inference/bin/python
import faiss
import pandas as pd
import numpy as np

data_fin = pd.read_csv('../../data_fin.tsv',sep='\t')

query_zpids = data_fin.sample(n = 10)

all_embs = []
for row in data_fin.iterrows():
    all_embs+= [row[1]['itemsage_emb_ep8']]
all_embs = np.array(all_embs)
all_embs.shape


query_embs = []
for row in query_zpids.iterrows():
    query_embs+= [row[1]['itemsage_emb_ep8']]
query_embs = np.array(query_embs)
print('Done')