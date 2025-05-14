import torch
import json
import requests
# import subprocess
import os
import tarfile 
from transformers import AutoModel, AutoTokenizer
from torch import nn
from tqdm import tqdm
from datetime import datetime
import urllib, urllib.request


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

def get_arxiv_ids_from_bbl(path):
    bib_list = []
    with open(path, "r") as bib:
        for line in bib.readlines():
            bib_list.append(line.rstrip())
    
    ids = []
    for i, line in enumerate(bib_list):
        # identify new entry
        if line.startswith("\\bibitem"):
            # identify title
            def get_next_newblock_lines(current_index, current_lines=[]):
                if len(current_lines) == 0:
                    if bib_list[current_index+1].startswith("\\newblock"): # add first line of newblock
                        # continue until end of newblock
                        return get_next_newblock_lines(current_index+1, [bib_list[current_index+1]])
                    else: # continue to search for start of newblock
                        return get_next_newblock_lines(current_index+1, [])
                else: # continue until end of newblock
                    running_index = current_index+1
                    while not bib_list[running_index].startswith("\\"):
                        current_lines.append(bib_list[running_index])
                        running_index += 1

                return current_lines

            title_lines = get_next_newblock_lines(i)
            title = ""
            for title_idx in range(len(title_lines)):
                if title_idx == 0:
                    title += title_lines[title_idx].lstrip("\\newblock ")
                else:
                    title += title_lines[title_idx].lstrip(" ")
            title = title.rstrip(".")

            # use title to search for and retrieve the arxiv id
            title_for_url = title.replace(" ", "+")
            url = f'https://export.arxiv.org/api/query?search_query={title_for_url}&searchtype=title&start=0&max_results=1'
            data = urllib.request.urlopen(url)
            xml = data.read().decode('utf-8')
            try: # skip to the actual search result
                xml = xml.split("<entry>")[1]
            except IndexError:
                continue # skip this bib item if the search does not yield a result
            xml = xml.replace(">\n", ">SPLIT").split("SPLIT")
            xml = [# cleanup whitespace and linebraks
                item.lstrip().replace("\n", " ")
                for item in xml
            ]
            # filter relevant elements and map them
            xml_dict = {}
            for item in xml:
                for label in ["<id>", "<title>", "<summary>", "<name>"]:
                    if item.startswith(label):
                        clean_label = label.lstrip("<").rstrip(">")
                        item_body = item.lstrip(label).rstrip(label.replace("<", "</"))
                        xml_dict[clean_label] = item_body

            # compare titles
            t1 = title.lower().replace(" ", "")
            t2 = xml_dict["title"].lower().replace(" ", "")
            if not t1 == t2:
                # print(f"{t1} != {t2} ?!")
                continue
            else:
                # print("titles match, yay!")
                pass

            # single out the arxiv id
            arxiv_id = xml_dict["id"].split("/")[-1]
            ids.append(arxiv_id)

    # print(ids)
    return ids


def get_bbl_path_from_arxiv_id(id):
    path = f"data/{id}/"
    for _, _, files in os.walk(f"data/{id}"):
        matches = [file for file in files if file.endswith(".bbl")]
        assert len(matches) == 1
        path += matches[0]
        break
    return path


def locate_main_tex_file(id):
    path = f"data/{id}/"
    for _, _, files in os.walk(f"data/{id}"):
        matches = [file for file in files if file.endswith(".tex")]
        # TODO
        break


def extract_full_latex_textbody(id):
    # TODO
    pass
    

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
    #     json.dump({
    #         "time": f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    #         "sim": f"{sim}"
    #     }, f, ensure_ascii=False, indent=4)

    for id in ["1607.06450", "1409.0473", "1703.03906"]:
        # dl_arxiv(id)
        path = get_bbl_path_from_arxiv_id(id)
        get_arxiv_ids_from_bbl(path)