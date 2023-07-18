import os
import openai
import sys
import tiktoken
from openai.embeddings_utils import get_embedding
import pandas as pd
import numpy as np
# would need to do this in server
# openai.api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhIjoie1wiaWRcIjpcIjAwdTFuN2phbTloTThxMjh5MWQ4XCIsXCJuYW1lXCI6XCJTaHJleWFzIFZlcm1hXCIsXCJlbWFpbFwiOlwic2hyZXlhc3ZlQHppbGxvd2dyb3VwLmNvbVwiLFwiaW1hZ2VcIjpcIlwiLFwiZ3JvdXBzXCI6W1wiemdhaV9sbG1fdXNlcnNfc3RhZ2VcIl19IiwiaWF0IjoxNjg4NzU5MjgwLCJleHAiOjE2ODkzNjQwODB9.QZTWmp3JY8nJ6AV8snMOgWb8oNfRa-qRaLbyu3qpR-k'
# openai.api_base = "https://zgai-llm-api.int.stage-k8s.zg-aip.net/openai/v1"

import tensorflow as tf
import tensorflow_hub as hub

# text = "Rare opportunity to own one of 14 Cobblestone Cottages. Come lock yourself away in the beautiful, gated community, experience the serene & peaceful environment. Captivating residential oasis located in the heart of Everett, this property offers a unique blend of comfort, convenience, and style, providing an ideal home for those seeking a modern urban lifestyle.   Craftsman Style stand-alone cottage offers wood beam design, vaulted ceilings, two Primary bedrooms, office loft, granite countertops, stainless-steel appliances, ceiling fans, skylights, gas fireplace, newly finished hardwoods & large, covered porch. The natural light floods through this home's expansive windows, creating a bright and inviting atmosphere. Seller related to Broker."
# embedding = openai.Embedding.create(
#     input=text, model="text-embedding-ada-002"
# )["data"][0]["embedding"]
# len(embedding)


df = pd.read_csv('../../zpid_listingDesc.tsv',sep='\t')

# embedding_model = "text-embedding-ada-002"
# embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# encoding = tiktoken.get_encoding(embedding_encoding)
df = df[df['listingDescription'].isna()==False]
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)
# df["n_tokens"] = df['listingDescription'].apply(lambda x: len(encoding.encode(x)))
log_interval = 1000
with open('../../text_embeddings_30day_LB_USE.tsv','w') as f :
    data = ''
    for idx,row in enumerate(df.iterrows()):
        # embedding = openai.Embedding.create(input=row[1].listingDescription,model=embedding_model)["data"][0]["embedding"]
        embedding = embed([row[1].listingDescription])[0,:].numpy().tolist()
        f.write('{}\t{}\n'.format(row[1].zpid,embedding))
        if (idx%1000 == 0) or (idx+1 == df.shape[0]):
            print('DONE : {:.2f} %.'.format(100.*(idx+1)/df.shape[0]))
    f.close()
