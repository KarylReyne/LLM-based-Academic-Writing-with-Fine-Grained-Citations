# from: https://huggingface.co/reasonir/ReasonIR-8B
# dataset: https://huggingface.co/datasets/reasonir/reasonir-data

from transformers import AutoModel, AutoTokenizer
from torch import nn
model = AutoModel.from_pretrained("reasonir/ReasonIR-8B", torch_dtype="auto", trust_remote_code=True)

query = "The quick brown fox jumps over the lazy dog."
document = "The quick brown fox jumps over the lazy dog."
query_instruction = ""
doc_instruction = ""

# model= nn.DataParallel(model)
model = model.to("cuda")
model.eval()

query_emb = model.encode(query, instruction=query_instruction)
doc_emb = model.encode(document, instruction=doc_instruction)
sim = query_emb @ doc_emb.T

import json
with open('out/test_sim.json', 'w', encoding='utf-8') as f:
    json.dump(sim, f, ensure_ascii=False, indent=4)
