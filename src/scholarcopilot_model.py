from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import faiss
import numpy as np
import h5py
import json
from tqdm import tqdm
import os
import sys
import glob
import re
import time
from latex_parsing import *
import ijson


def retrieve_reference(index, lookup_indices, cite_start_hidden_state, top_k=5):
    start = time.time()
    print("Retrieving reference")

    if isinstance(cite_start_hidden_state, torch.Tensor):
        cite_start_hidden_state = cite_start_hidden_state.cpu().numpy()

    if cite_start_hidden_state.ndim == 1:
        cite_start_hidden_state = cite_start_hidden_state.reshape(1, -1)

    faiss.normalize_L2(cite_start_hidden_state)

    # this retrieves papers by comparing the cite token embedding to the embedded corpus documents (title+abstract)
    distances, indices = index.search(cite_start_hidden_state, top_k)
    retrieved_corpus_indices = []

    for i in indices[0]:
        each_index = str(lookup_indices[i], 'ascii')
        print("index is ", each_index)
        retrieved_corpus_indices.append(each_index)

    print("retrieved_corpus_indices", retrieved_corpus_indices)
    print("distances[0]", distances[0])
    print("***************Retrieve cost time: ", time.time() - start)
    assert len(retrieved_corpus_indices) == len(distances[0])
    return list(zip(retrieved_corpus_indices, distances[0]))


def single_complete_step(model, tokenizer, device, input_text):
    print("completing sentence ...\n")
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    if len(inputs.input_ids[0]) > 15000:
        return input_text, None
    stop_token_ids = tokenizer.convert_tokens_to_ids(['<|cite_start|>', '<|paper_end|>'])
    # print("stop_token_ids", stop_token_ids)
    eos_token_id = stop_token_ids[0]

    max_new_tokens = 4096
    try: # terminate early if process runs out of memory
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.6,
                eos_token_id=eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        
        generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=False)

        new_input = tokenizer(generated_text, return_tensors="pt").to(device)
        with torch.no_grad():
            new_output = model(
                new_input.input_ids,
                attention_mask=new_input.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        cite_rep = new_output.hidden_states[-1][:, -1, :]
    except torch.OutOfMemoryError:
            print(f"CUDA out of memory. Terminating generation early at length {len(inputs.input_ids[0])}/15000")
            return input_text, None

    new_content = generated_text
    if "<|paper_end|>" in new_content:
        end_index = new_content.index("<|paper_end|>")
        return generated_text[:end_index + len("<|paper_end|>")], None

    return new_content, cite_rep


def clean_latex_text(input_text):
    # Remove document class, packages, makefile and document tags
    patterns = [
        r'\\documentclass\{[^}]*\}',
        r'\\usepackage\{[^}]*\}',
        r'\\makefile',
        r'\\begin\{document\}',
        r'\\end\{document\}'
    ]

    result = input_text
    for pattern in patterns:
        result = re.sub(pattern, '', result)

    return "<|paper_start|> " + result.strip()


def preprocess_input_text(input_text):
    input_text = clean_latex_text(input_text)
    # print("preprocess_input_text result", input_text)
    return input_text


def cut_after_third_sentence(text, num_sentences=3):
    count = 0
    # print("num_sentences", num_sentences)
    text, citations = down_sample_cut(text)
    # print("down_sample_cut(text) result", text, citations)
    for i in range(len(text) - 1):
        if text[i] in ['.', '!', '?'] and (text[i + 1] == ' ' or text[i + 1] == '\n'):
            count += 1
            if count == num_sentences:
                return True, up_sample_cut(text[:i + 1], citations)
    return False, up_sample_cut(text, citations)


def down_sample_cut(input_text):
    pattern = r'<\|cite_start\|>(.*?)<\|cite_end\|>'
    citations = re.findall(pattern, input_text)

    output_text = input_text
    for i in range(len(citations)):
        output_text = re.sub(pattern, f'${i}$', output_text, count=1)

    return output_text, citations


def up_sample_cut(input_text, citation_list):
    for i in range(len(citation_list)):
        if f"${i}$" in input_text:
            input_text = input_text.replace(f"${i}$", "<|cite_start|>" + citation_list[i] + "<|cite_end|>")
    return input_text


def collect_retrieval_results(retrieved_k_results, retrieval_dataset):
    references = []
    distances = []
    for each in retrieved_k_results:
        curr_corpus_idx, distance = each
        if curr_corpus_idx not in retrieval_dataset:
            print(f"index {curr_corpus_idx} not found in retrieval_dataset")
            continue
        references.append(retrieval_dataset[curr_corpus_idx])
        distances.append(distance)

    print(f"best reference before passage retrieval: {references[0]["arxiv_id"]}")
    return references, distances


def replace_citations(current_text, unique_reference_id_list, retrieval_dataset, arxiv_to_corpus_id_map):
    print("IN replace_citations\n")
    # Find all citations with pattern <|cite_start|>XXX<|cite_end|>
    pattern = r'<\|cite_start\|>(.*?)<\|cite_end\|>'
    # Keep track of current citation index
    citation_index = 0
    res_citation_data_list = []
    last_replacement = ""

    # Function to replace each match with corresponding reference id
    def replace_match(match):
        nonlocal citation_index, res_citation_data_list, last_replacement
        if citation_index < len(unique_reference_id_list):
            
            arxiv_id = unique_reference_id_list[citation_index][0]
            unique_id_suffix = unique_reference_id_list[citation_index][1]
            corpus_id = arxiv_to_corpus_id_map[arxiv_id]

            citation_key = f"arxivID-{arxiv_id}-{unique_id_suffix}"
            # print("citation_key", citation_key)
            replacement = "\\cite{" + citation_key + "}"

            citation_data = {
                "corpus_id": corpus_id,
                "arxiv_id": arxiv_id,
                "title": retrieval_dataset[corpus_id]["title"],
                "citation_key": citation_key
            }
            # print("citation_data", citation_data)
            if last_replacement == replacement:
                replacement = ""
            else:
                res_citation_data_list.append(citation_data)
                last_replacement = replacement
            citation_index += 1
            return replacement
        return match.group(0), res_citation_data_list  # Keep original if no more reference ids

    # Replace all citations
    result = re.sub(pattern, replace_match, current_text)
    result = result.replace("<|paper_start|> ", "").replace("<|cite_start|>", "")
    # print("res_citation_data_list", res_citation_data_list)

    return result, res_citation_data_list


def post_process_output_text(res_text, reference_arxiv_id_list, retrieval_dataset, arxiv_to_corpus_id_map):
    # print("post_process_output_text, res_text", res_text)
    output_text, citation_info_list = replace_citations(res_text, reference_arxiv_id_list, retrieval_dataset, arxiv_to_corpus_id_map)
    # print("post_process_output_text, citation_info_list ", citation_info_list)
    output_text = output_text.replace("<|paper_start|> ", "").replace(" <|paper_end|>", " <|section_end|>")
    # output_text = output_text.replace("<|paper_start|> ", "")
    output_text = merge_consecutive_citations(output_text)
    return output_text, citation_info_list


def merge_consecutive_citations(text):
    pattern = r'\\cite\{[^}]+\}(\s*\\cite\{[^}]+\})*'

    def merge_group(match):
        group_text = match.group(0)
        cite_pattern = r'\\cite\{([^}]+)\}'
        keys = re.findall(cite_pattern, group_text)
        return "\\cite{" + ", ".join(keys) + "}"

    result = re.sub(pattern, merge_group, text)

    return result


def load_retrieval_dataset(retrieval_dataset_path, complete_dataset_path, id_map):
    print("loading retrieval dataset...")
    retrieval_dataset = {}
    print()
    counter = 0
    try:
        with open(retrieval_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")
                retrieval_dataset[item["corpus_id"]] = item
                counter += 1
    except FileNotFoundError:
        with open(complete_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")
                arxiv_id = item["arxiv_id"]
                corpus_id = id_map[arxiv_id]
                rec = {
                    "corpus_id": corpus_id,
                    "arxiv_id": arxiv_id,
                    "title": item["title"], 
                    "abstract": " ".join(item["abstract"]), 
                    "sections": item["sections"]
                }
                retrieval_dataset[corpus_id] = rec
                counter += 1
                with open(retrieval_dataset_path, "a") as outfile:
                    json.dump(rec, outfile)
                    outfile.write("\n")
    print("retrieval dataset loaded.")
    return retrieval_dataset


def arxiv_to_corpus_id(path, processed_corpus_path):
    print("loading arxiv_to_corpus_id map...")
    id_map = {}
    try:
        with open(path, "r") as mapfile:
            id_map = json.load(mapfile)
    except FileNotFoundError:
        counter = 0
        print()
        with open(processed_corpus_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")
                id_map[item["arxiv_id"]] = item["corpus_id"]
                counter += 1
        with open(path, "w") as file:
            json.dump(id_map, file, ensure_ascii=False, indent=4)
    print("arxiv_to_corpus_id map loaded.")
    return id_map


def load_corpus_base(corpus_dir="../embedded_corpus/1128_shards/"):
    encoded_list = []
    lookup_indices_list = []

    h5_files = sorted(glob.glob(os.path.join(corpus_dir, "*.h5")))

    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {corpus_dir}")

    print(f"Loading embedded vectors. Found {len(h5_files)} files to load")

    for file_path in h5_files:
        try:
            with h5py.File(file_path, 'r') as f:
                encoded_list.append(f['encoded'][:])
                lookup_indices_list.append(f['lookup_indices'][:])
            print(f"Successfully loaded {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if encoded_list and lookup_indices_list:
        encoded = np.concatenate(encoded_list, axis=0)
        lookup_indices = np.concatenate(lookup_indices_list, axis=0)
        print(f"Combined shape - encoded: {encoded.shape}, lookup_indices: {lookup_indices.shape}")
        print("embedded vectors loaded.")
        return encoded, lookup_indices
    else:
        raise ValueError("No data was successfully loaded")


def load_faiss_index(index_dir):
    index = faiss.read_index(os.path.join(index_dir, 'index'))
    with open(os.path.join(index_dir, 'lookup_indices.npy'), 'rb') as f:
        lookup_indices = np.load(f, allow_pickle=True)
    return index, lookup_indices


def load_model(model_path, device):
    config = AutoConfig.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    special_tokens = ['<|paper_start|>', '<|paper_end|>', '<|cite_start|>', '<|cite_end|>', '<|reference_start|>',
                      '<|reference_end|>']
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print("model loaded successfully")
    return model, tokenizer




