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


ARXIV_MAX_RESULTS = 20
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


def get_citations_data_from_bbl(path, additional_citation_records=None, force_download=False):
    source_id = path.split("/")[1]
    data_path = f"data/citations_data/cd_source-{source_id}.json"

    # determine whether to download or load from disk
    file_found = None
    try:
        with open(data_path) as f:
            pass
        file_found = True
    except FileNotFoundError:
        file_found = False

    if not force_download and file_found:
        with open(data_path, "r") as f:
            citations_data = json.load(f)
        ids = [x[1]["arxiv_id"] for x in citations_data.items()]
    else: # download
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
                url = f'https://export.arxiv.org/api/query?search_query={title_for_url}&searchtype=title&start=0&max_results={ARXIV_MAX_RESULTS}'
                data = urllib.request.urlopen(url)
                xml = data.read().decode('utf-8')
                try: # skip to the actual search results
                    xml = xml.split("<entry>")[1:]
                except IndexError:
                    continue # skip this bib item if the search does not yield a result

                # extract and store information from search result entries
                xml_dict = {"bib_id": f"{bib_id}"}
                for search_result_entry in xml:
                    entry = [ # cleanup whitespace and linebraks
                        item.lstrip().replace("\n", " ")
                        for item in search_result_entry.replace(">\n", ">SPLIT").split("SPLIT")
                    ]

                    # get the title of this entry, if possible
                    entry_title = ""
                    for entry_item in entry:
                        if entry_item.startswith("<title>"):
                            entry_title = entry_item.lstrip("<title>").rstrip("</title>")
                            break

                    # check if the entry is about the correct paper by comparing titles
                    process_title = lambda t: t.lower().replace(" ", "")
                    t1 = process_title(title) # bib entry
                    t2 = process_title(entry_title) # search result entry
                    if not t1 == t2:
                        # print(f"t1: {t1}")
                        # print(f"t2: {t2}")
                        continue

                    # record useful information in the citation record aka xml_dict
                    for entry_item in entry:
                        for label in ["<id>", "<title>", "<summary>", "<name>"]:
                            if entry_item.startswith(label):
                                clean_label = label.lstrip("<").rstrip(">")
                                item_body = entry_item.lstrip(label).rstrip(label.replace("<", "</"))
                                if clean_label == "name":
                                    try: # check if label already exists (for multiple author names)
                                        xml_dict[clean_label] += f" {item_body}"
                                    except KeyError:
                                        xml_dict[clean_label] = item_body
                                elif clean_label == "id":
                                    xml_dict["arxiv_id"] = item_body.split("/")[-1]
                                else:
                                    xml_dict[clean_label] = item_body

                # check if the paper was found via arxiv search
                if xml_dict == {"bib_id": f"{bib_id}"}:
                    # TODO: maybe get the latex paper from somewhere else?
                    continue

                # store this citation (xml_dict) in citation_data
                citations_data[bib_id] = xml_dict

                # add id for recursive paper discovery
                ids.append(xml_dict["arxiv_id"])

    # incorporate additional records
    if additional_citation_records is not None:
        for additional_record in additional_citation_records.items():
            citations_data[additional_record[0]] = additional_record[1]
            ids.append(additional_record[1]["arxiv_id"])

    # save citations_data to disk
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(citations_data, f, ensure_ascii=False, indent=4)

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
    if not path.endswith(".tex"):
        path += ".tex"
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
                    input_file = line.split("{")[1].split("}")[0]
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
    
def get_target_sections(target_doc, section_labels=["\\section", "\\subsection"]):
    pass


def get_source_citations(source_id, target_citation_record):
    target_bib_id = target_citation_record["bib_id"]
    target_arxiv_id = target_citation_record["arxiv_id"]

    source_doc, _ = extract_full_latex_textbody(source_id)
    target_doc = None
    try:
        target_doc, _ = extract_full_latex_textbody(target_arxiv_id)
    except AssertionError: # download if target not found
        dl_arxiv(target_arxiv_id)
        target_doc, _ = extract_full_latex_textbody(target_arxiv_id)
    assert target_doc != None

    citing_sents = identify_citing_sentences(source_doc, target_bib_id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER)
    tokens = tokenizer(target_doc).to("cuda")
    tokens = tokens["input_ids"] # get only the encoded tokens
    target_doc_sections = [tokenizer.decode(tokens[i:i+CHUNK_SIZE]) for i in range(0, len(tokens), CHUNK_SIZE)]

    return citing_sents, target_doc_sections


# srun --job-name "ReasonIRtest" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:2 --time 1:00:00 --pty bash
# cd src
# conda activate citations
if __name__ == "__main__":

    # for id in ["1607.06450"]:# , "1409.0473", "1703.03906"]:
    #     # dl_arxiv(id)
    #     path = get_bbl_path_from_arxiv_id(id)
    #     citations_data, ids = get_citations_data_from_bbl(path)
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
    #     #         _, _new_ids = get_citations_data_from_bbl(_path)
    #     #         [new_ids.append(y) for y in _new_ids]

    #     #     ids_to_process = new_ids
    #     #     new_ids = []
    #     #     recursion_depth -= 1


    id = "2108.09084" # Fastformer
    path = get_bbl_path_from_arxiv_id(id)
    citations_data, _ = get_citations_data_from_bbl(path, additional_citation_records={
        "vaswani2017attention": {
            "bib_id": "vaswani2017attention",
            "arxiv_id": "1706.03762", 
            "title": "Attention is all you need", 
            "summary": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.", 
            "name": "Ashish Vaswani Noam Shazeer Niki Parmar Jakob Uszkoreit Llion Jones Aidan~N Gomez Lukasz Kaiser Illia Polosukhin"
        }
    })
    print(len(citations_data.items()))

    target_bib_id = "vaswani2017attention" # Transformer
    target_citation_record = citations_data[target_bib_id]

    # DEBUG
    exit

    citing_sents, target_doc_sections = get_source_citations(id, target_citation_record)

    model = AutoModel.from_pretrained(MODEL_IDENTIFIER, torch_dtype="auto", trust_remote_code=True)
    model = model.to("cuda")
    model.eval()

    citing_sent = citing_sents[0]
    similarity_records = {}
    idx = 0
    for candidate_chunk in target_doc_sections:
        query_instruction = "" 
        doc_instruction = ""

        # TODOs
        # experiment with instructions, specify the mask token
        # create map fastformer 2108.09084 to transformer 1706.03762
        # test query context length
        # use sections/subsections instead of chunks
        # replace figure with captions text
        # 
        # https://github.com/stanfordnlp/stanza
        # https://github.com/IllDepence/unarXive?tab=readme-ov-file
        # https://huggingface.co/datasets/howey/unarXive
        # 

        query_emb = model.encode(citing_sent, instruction=query_instruction)
        doc_emb = model.encode(candidate_chunk, instruction=doc_instruction)
        sim = query_emb @ doc_emb.T

        similarity_records[f"chunk-{idx}"] = {
            "query": citing_sent,
            "chunk": candidate_chunk,
            "sim": f"{sim}"
        }
        
        idx += 1

    similarity_records = dict(sorted(similarity_records.items(), key=lambda item: item[1]["sim"], reverse=True))

    with open('out/similarity_records.json', 'w', encoding='utf-8') as f:
        json.dump({
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}": {
                "query_doc_id": f"{id}",
                "target_doc_id": f"{target_citation_record["arxiv_id"]}",
                "records": similarity_records
            }
        }, f, ensure_ascii=False, indent=4)