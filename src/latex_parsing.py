import torch
import json
import requests
import os
import tarfile
import nltk
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch import nn
from tqdm import tqdm
from datetime import datetime
import urllib, urllib.request


ARXIV_MAX_RESULTS = 20
CITATION_MASK = "<CIT-MASK>"
LABEL_SEPARATOR = "<LABEL-SEP>"
TOKENIZER_BEGIN_TOKEN = "<|begin_of_text|>"


def download_from_arxiv(id='1706.03762'):
    url = f'https://arxiv.org/src/{id}'
    response = requests.get(url)

    # download source archive
    with open(f"data/{id}.tar.gz", "wb") as handle:
        for data in tqdm(response.iter_content(chunk_size=1024), unit="kB"):
            handle.write(data)
        handle.close()
    try:
        os.makedirs(f"data/{id}")
    except FileExistsError:
        pass

    # extract archive
    try:
        tar = tarfile.open(f"data/{id}.tar.gz")
        tar.extractall(f"data/{id}", filter="tar")
    except tarfile.ReadError as e:
        print(f"data/{id}.tar.gz could not be extracted successfully.")
        raise e

    # delete archive
    os.remove(f"data/{id}.tar.gz")
    assert not os.path.isfile(f"data/{id}.tar.gz")


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


def search_arxiv_for_citations_data(id, additional_citation_records=None, force_download=False):
    path = get_bbl_path_from_arxiv_id(id)
    data_path = f"data/citations_data/cd_source-{id}.json"

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


def identify_citations_in_source_doc(source_doc, bib_id, tokenizer, query_expansion_method="left", query_context_size=128):
    citing_sents = []
    citing_context = []

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
        citing_sents = identify_citations_in_source_doc(source_doc, bib_id, tokenizer, query_expansion_method, query_context_size)

    # prepare tokens for query expansion
    tokenizer.add_special_tokens({"additional_special_tokens": [bib_id]}) # ensure that the tokenizer does not destroy the bib id
    tokens = tokenizer(source_doc).to("cuda")
    tokens = tokens["input_ids"] # get only the encoded tokens
    indices = [] # indices of each bib id in the source doc
    for i, e in enumerate(tokens):
        t = tokenizer.decode(e)
        if t == bib_id:
            indices.append(i)

    # get larger context if required
    if query_expansion_method == "center":
        w = int(np.floor(query_context_size/2)) # context window size
        for index in indices:
            low = max(index-w, 0)
            high = min(index+w, len(tokens)-1)
            context = tokenizer.decode(tokens[low:high])
            context = context.replace(TOKENIZER_BEGIN_TOKEN, "")
            context = context.replace(bib_id, CITATION_MASK)
            citing_context.append(context)

        assert len(citing_sents) == len(citing_context)

    elif query_expansion_method == "left":
        for index in indices:
            low = max(index-query_context_size, 0)
            high = index
            context = tokenizer.decode(tokens[low:high])
            context = context.replace(TOKENIZER_BEGIN_TOKEN, "")
            context = context.replace(bib_id, CITATION_MASK)
            citing_context.append(context)

        assert len(citing_sents) == len(citing_context)

    else:
        raise NotImplementedError(f"Query expansion method '{query_expansion_method}' is not implemented. Currently supported are 'center' and 'left'.")
        
    return (citing_sents, citing_context)


def get_target_sections(
    target_doc_lines, 
    tokenizer, 
    section_labels=["\\begin{abstract}", "\\section", "\\subsection"], 
    with_chunking=False,
    chunk_size=512
):
    
    def is_section_start(l):
        b = False
        for s in section_labels:
            b = b or l.startswith(s)
        return b
    

    sections = []

    in_section = False
    current_section = ""
    current_section_label = None
    for line in target_doc_lines:
        line = line.rstrip("\n")

        if is_section_start(line):
            if in_section: # terminate the current section

                if with_chunking: # chop section into chunks
                    current_section = current_section.split(LABEL_SEPARATOR)[1] # remove section label
                    tokens = tokenizer(current_section).to("cuda")
                    tokens = tokens["input_ids"] # get only the encoded tokens

                    chunk_idx = 0
                    for i in range(0, len(tokens), chunk_size):
                        chunk_label = f"{current_section_label}-{chunk_idx}"
                        chunk = tokenizer.decode(tokens[i:i+chunk_size]).replace(TOKENIZER_BEGIN_TOKEN, "")
                        sections.append(chunk_label+LABEL_SEPARATOR+chunk)
                        chunk_idx += 1
                else:
                    sections.append(current_section)
                in_section = False

            # start collecting the new section
            current_section_label = line.split("}")[0].split("{")[1]
            current_section = current_section_label+LABEL_SEPARATOR+line.split("}")[1]
            in_section = True
        else:
            if in_section: # expand current section
                if not line.startswith("%"): # ignore comments
                    current_section += " "+line
            else: # don't do anything
                continue

    return sections


def get_source_citations(id, target_citation_record, tokenizer, with_chunking=False, chunk_size=512, query_expansion_method="left", query_context_size=128):
    target_bib_id = target_citation_record["bib_id"]
    target_arxiv_id = target_citation_record["arxiv_id"]

    source_doc, _ = extract_full_latex_textbody(id)
    target_doc_lines = None
    try:
        _, target_doc_lines = extract_full_latex_textbody(target_arxiv_id)
    except AssertionError: # download if target not found
        download_from_arxiv(target_arxiv_id)
        _, target_doc_lines = extract_full_latex_textbody(target_arxiv_id)
    assert target_doc_lines != None

    # list of (citing sentence, larger context centered at citation)
    source_doc_citations = identify_citations_in_source_doc(source_doc, target_bib_id, tokenizer, query_expansion_method, query_context_size)
    target_doc_sections = get_target_sections(target_doc_lines, tokenizer, with_chunking=with_chunking, chunk_size=chunk_size)

    return source_doc_citations, target_doc_sections
