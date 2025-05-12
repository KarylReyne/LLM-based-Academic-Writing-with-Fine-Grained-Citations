import torch
import json
import requests
# import subprocess
import os
import tarfile 
from transformers import AutoModel, AutoTokenizer
from torch import nn
from tqdm import tqdm


def dl_arxiv(id='1706.03762'):
    url = f'https://arxiv.org/src/{id}'
    
    response = requests.get(url)
    with open(f"data/{id}.tar.gz", "wb") as handle:
        for data in tqdm(response.iter_content(chunk_size=1024), unit="kB"):
            handle.write(data)
        handle.close()
    try:
        os.makedirs(f"data/{id}")
    except FileExistsError:
        pass

    tar = tarfile.open(f"data/{id}.tar.gz")
    tar.extractall(f"data/{id}", filter="tar")
    

# srun --job-name "ReasonIRtest" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:2 --time 1:00:00 --pty bash
# cd src
# conda activate citations
if __name__ == "__main__":

    # # from: https://huggingface.co/reasonir/ReasonIR-8B
    # # dataset: https://huggingface.co/datasets/reasonir/reasonir-data
    # model = AutoModel.from_pretrained("reasonir/ReasonIR-8B", torch_dtype="auto", trust_remote_code=True)

    # query = "The quick brown fox jumps over the lazy dog."
    # document = "The fast brown fox jumps over the lazy dog."
    # query_instruction = ""
    # doc_instruction = ""

    # # print(torch.cuda.device_count())
    # # model= nn.DataParallel(model)
    # model = model.to("cuda")
    # model.eval()

    # query_emb = model.encode(query, instruction=query_instruction)
    # doc_emb = model.encode(document, instruction=doc_instruction)
    # sim = query_emb @ doc_emb.T

    # with open('out/test_sim.json', 'w', encoding='utf-8') as f:
    #     json.dump({"sim": f"{sim}"}, f, ensure_ascii=False, indent=4)

    dl_arxiv()