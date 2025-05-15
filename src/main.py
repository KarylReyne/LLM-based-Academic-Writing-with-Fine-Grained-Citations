import torch
import json
import requests
import os
import tarfile
import nltk
from transformers import AutoModel, AutoTokenizer
from torch import nn
from tqdm import tqdm
from datetime import datetime
import urllib, urllib.request


CITATION_MASK = "<CIT-MASK>"
MODEL_IDENTIFIER = "reasonir/ReasonIR-8B"
CHUNK_SIZE = 128


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

    try:
        tar = tarfile.open(f"data/{id}.tar.gz")
        tar.extractall(f"data/{id}", filter="tar")
    except tarfile.ReadError as e:
        print(f"data/{id}.tar.gz could not be extracted successfully.")
        raise e


def get_next_block_lines(current_index, bib_list, block_label, current_lines=[]):
    if len(current_lines) == 0:
        if bib_list[current_index+1].startswith(block_label): # add first line of block
            # continue until end of block
            return get_next_block_lines(current_index+1, bib_list, block_label, [bib_list[current_index+1]])
        else: # continue to search for start of block
            return get_next_block_lines(current_index+1, bib_list, block_label, [])

    else: # continue until end of block
        running_index = current_index+1
        while not bib_list[running_index].startswith("\\"):
            current_lines.append(bib_list[running_index])
            running_index += 1

    return current_lines


def get_citation_data_from_bbl(path):
    bib_list = []
    with open(path, "r") as bib:
        for line in bib.readlines():
            bib_list.append(line.rstrip())
    
    citations_data = {}
    ids = []
    for i, line in enumerate(bib_list):
        # identify new entry
        if line.startswith("\\bibitem"):

            # identify bib id
            bibitem_lines = get_next_block_lines(i-1, bib_list, "\\bibitem")
            bib_id = ""
            for l in bibitem_lines:
                bib_id += l
            bib_id = bib_id.split("]{")[-1].split("}")[0]

            # identify title
            title_lines = get_next_block_lines(i, bib_list, "\\newblock")
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
            xml_dict = {"bib_id": bib_id}
            for item in xml:
                for label in ["<id>", "<title>", "<summary>", "<name>"]:
                    if item.startswith(label):
                        clean_label = label.lstrip("<").rstrip(">")
                        item_body = item.lstrip(label).rstrip(label.replace("<", "</"))
                        if clean_label == "name":
                            try: # check if entry already exists (for multiple author names)
                                xml_dict[clean_label] += f" {item_body}"
                            except KeyError:
                                xml_dict[clean_label] = item_body
                        elif clean_label == "id":
                            xml_dict["arxiv_id"] = item_body.split("/")[-1]
                        else:
                            xml_dict[clean_label] = item_body

            # compare titles
            t1 = title.lower().replace(" ", "")
            t2 = xml_dict["title"].lower().replace(" ", "")
            if not t1 == t2:
                # print(f"{t1} != {t2} ?!")
                # do not add this citation entry to citation_data bc the arxiv id could not be determined
                # TODO: maybe get the latex paper from somewhere else?
                continue 
            else:
                # print("titles match, yay!")
                pass

            # store this citation (xml_dict) in citation_data
            citations_data[bib_id] = xml_dict

            # add id for recursive paper discovery
            ids.append(xml_dict["arxiv_id"])

    return citations_data, ids


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
        for match in matches:
            match_lines = []
            with open(f"data/{id}/{match}", "r") as tex:
                for line in tex.readlines():
                    if line.startswith("\\documentclass"):
                        path += match
                        break # lines loop
        if path != f"data/{id}/": # found the main tex file
            break # matches loop
        else: # check the next tex file
            continue
    try:
        assert path != f"data/{id}/"
    except AssertionError as e:
        print(f"Main tex file for document {id} not found!")
        raise e
    return path


def extract_input_latex(path):
    lines = []
    with open(path, "r") as tex:
        for line in tex.readlines():
            if not line.startswith("\n"): # removes linebreaks
                lines.append(line)
    return lines


def extract_full_latex_textbody(id):
    path = locate_main_tex_file(id)
    lines = []
    collect_lines = False
    with open(path, "r") as tex:
        for line in tex.readlines():
            
            if line.startswith("\\begin{document}"):
                collect_lines = True
            if line.startswith("\\end{document}"):
                collect_lines = False

            if collect_lines and not line.startswith("\n"): # removes linebreaks

                if line.startswith("\\input"): # handle latex \input
                    input_file = line.lstrip("\\input{").split("}")[0]
                    input_path = f"data/{id}/{input_file}"
                    input_lines = extract_input_latex(input_path)
                    for input_line in input_lines:
                        lines.append(input_line)
                else:
                    lines.append(line)
    
    return " ".join(lines), lines


def identify_citing_sentences(source_doc, bib_id):
    citing_sents = []

    # split source doc into sentences and retain sentences that contain bib_id
    try:
        for sent in nltk.sent_tokenize(source_doc):
            if sent != sent.replace(bib_id, ""):
                masked_sent = ""
                parts = sent.split(bib_id)
                for i in range(len(parts)):
                    masked_sent += f"{CITATION_MASK}{parts[i]}"
                masked_sent = masked_sent.lstrip(CITATION_MASK)
                citing_sents.append(masked_sent)
    except LookupError:
        nltk.download('punkt_tab')
        citing_sents = identify_citing_sentences(source_doc, bib_id)
    
    return citing_sents
    

def get_source_citations(source_id, target_citation_record):
    target_bib_id = target_citation_record["bib_id"]
    target_arxiv_id = target_citation_record["arxiv_id"]

    source_doc, _ = extract_full_latex_textbody(source_id)
    target_doc, _ = extract_full_latex_textbody(target_arxiv_id)

    citing_sents = identify_citing_sentences(source_doc, target_bib_id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER)
    tokens = tokenizer(target_doc).to("cuda")
    tokens = tokens["input_ids"] # get only the encoded tokens
    target_doc_chunks = [tokenizer.decode(tokens[i:i+CHUNK_SIZE]) for i in range(len(tokens))]

    return citing_sents, target_doc_chunks


# srun --job-name "ReasonIRtest" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:2 --time 1:00:00 --pty bash
# cd src
# conda activate citations
if __name__ == "__main__":

    # for id in ["1607.06450"]:# , "1409.0473", "1703.03906"]:
    #     # dl_arxiv(id)
    #     path = get_bbl_path_from_arxiv_id(id)
    #     citations_data, ids = get_citation_data_from_bbl(path)
    #     # print(citations_data)

    #     # recursion_depth = 3
    #     # ids_to_process = ids
    #     # new_ids = []
    #     # while recursion_depth > 0:
    #     #     for x in ids_to_process:
    #     #         try:
    #     #             dl_arxiv(x)
    #     #         except tarfile.ReadError:
    #     #             continue # skip if extraction failed

    #     #         _path = get_bbl_path_from_arxiv_id(x)
    #     #         _, _new_ids = get_citation_data_from_bbl(_path)
    #     #         [new_ids.append(y) for y in _new_ids]

    #     #     ids_to_process = new_ids
    #     #     new_ids = []
    #     #     recursion_depth -= 1


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


    id = "1607.06450"
    path = get_bbl_path_from_arxiv_id(id)
    citations_data, _ = get_citation_data_from_bbl(path)
    target_bib_id = list(citations_data.keys())[0]
    target_citation_record = citations_data[target_bib_id]

    citing_sents, target_doc_chunks = get_source_citations(id, target_citation_record)

    model = AutoModel.from_pretrained(MODEL_IDENTIFIER, torch_dtype="auto", trust_remote_code=True)
    model = model.to("cuda")
    model.eval()

    citing_sent = citing_sents[0]
    similarity_records = {}
    idx = 0
    for candidate_chunk in target_doc_chunks:
        query_instruction = ""
        doc_instruction = ""

        query_emb = model.encode(citing_sent, instruction=query_instruction)
        doc_emb = model.encode(candidate_chunk, instruction=doc_instruction)
        sim = query_emb @ doc_emb.T

        similarity_records[f"chunk-{idx}"] = {
            "query": citing_sent,
            "chunk": candidate_chunk,
            "sim": f"{sim}"
        }
        
        idx += 1

    similarity_records = dict(sorted(similarity_records.items(), key=lambda item: item[1]["sim"]))

    with open('out/similarity_records.json', 'w', encoding='utf-8') as f:
        json.dump({
            "time": f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            "records": similarity_records
        }, f, ensure_ascii=False, indent=4)