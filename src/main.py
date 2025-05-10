# from: https://huggingface.co/reasonir/ReasonIR-8B
# dataset: https://huggingface.co/datasets/reasonir/reasonir-data

from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("reasonir/ReasonIR-8B", torch_dtype="auto", trust_remote_code=True)

query = "The quick brown fox jumps over the lazy dog."
document = "The quick brown fox jumps over the lazy dog."
query_instruction = ""
doc_instruction = ""
model = model.to("cuda")
model.eval()
query_emb = model.encode(query, instruction=query_instruction)
doc_emb = model.encode(document, instruction=doc_instruction)
sim = query_emb @ doc_emb.T
