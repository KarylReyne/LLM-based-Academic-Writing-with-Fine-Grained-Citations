from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import faiss
import numpy as np
import h5py
import os
import glob
import re
import time

from passage_retrieval_instructions import retrieval_instruction_query


def retrieve_reference(index, lookup_indices, cite_start_hidden_state, config, top_k=5, silent=False):
    start = time.time()
    if not silent:
        print("Retrieving reference")

    if isinstance(cite_start_hidden_state, torch.Tensor):
        cite_start_hidden_state = cite_start_hidden_state.cpu().numpy()

    if cite_start_hidden_state.ndim == 1:
        cite_start_hidden_state = cite_start_hidden_state.reshape(1, -1)

    faiss.normalize_L2(cite_start_hidden_state)

    # custom efSearch
    index.hnsw.efSearch = config["hnsw_efSearch"]

    # this retrieves papers by comparing the cite token embedding to the embedded corpus documents
    distances, indices = index.search(cite_start_hidden_state, top_k)
    retrieved_corpus_indices = []

    for i in indices[0]:
        each_index = str(lookup_indices[i], 'ascii')
        retrieved_corpus_indices.append(each_index)
    if not silent:
        print("retrieved_corpus_indices", retrieved_corpus_indices)
        print("distances[0]", distances[0])
        print("***************Retrieval cost (time): ", time.time() - start)

    return list(zip(retrieved_corpus_indices, distances[0]))


def single_complete_step(model, tokenizer, passage_retrieval_models, config, input_text, generation_breakpoint=15000, silent=False):
    if not silent:
        print("completing sentence ...\n")
    
    max_new_tokens = 4096
    try: # terminate early if process runs out of memory
        inputs = tokenizer(input_text, return_tensors="pt").to(config["scholarcopilot_device"])

        if len(inputs.input_ids[0]) > generation_breakpoint:
            return input_text, None

        stop_token_ids = tokenizer.convert_tokens_to_ids(['<|cite_start|>', '<|paper_end|>'])
        # print("stop_token_ids", stop_token_ids)
        eos_token_id = stop_token_ids[0]

        with torch.no_grad(): # generates until cite token
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

        if config["use_only_passage_retriever"]: # reasonir instead of sc retrieval
            with torch.no_grad():
                cite_rep = passage_retrieval_models["retriever"].encode(
                    generated_text,
                    instruction=retrieval_instruction_query,
                    # batch_size=config["retr_batch_size"],
                    # max_length=config["passage_length"]
                )
        else: # sc retrieval
            new_input = tokenizer(generated_text, return_tensors="pt").to(config["scholarcopilot_device"])
            with torch.no_grad(): # generates cite token representation
                new_output = model(
                    new_input.input_ids,
                    attention_mask=new_input.attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            cite_rep = new_output.hidden_states[-1][:, -1, :]
    except torch.OutOfMemoryError:
        if not silent:
            print(f"CUDA out of memory. Terminating generation early at length {len(inputs.input_ids[0])}/{generation_breakpoint}")
        return input_text, None

    new_content = generated_text
    if "<|paper_end|>" in new_content:
        end_index = new_content.index("<|paper_end|>")
        return generated_text[:end_index + len("<|paper_end|>")], None

    return new_content, cite_rep


def single_step_retrieval(text, index, lookup_indices, model, tokenizer, config, silent=False):
    new_input_text = text + " <|cite_start|>"
    new_input = tokenizer(new_input_text, return_tensors="pt").to(config["scholarcopilot_device"])
    try:
        with torch.no_grad():
            new_output = model(
                new_input.input_ids,
                attention_mask=new_input.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
    except torch.OutOfMemoryError:
        raise ScholarCopilotRetrievalError("raised ScholarCopilotRetrievalError due to running out of memory during forward pass")
    cite_rep = new_output.hidden_states[-1][:, -1, :]
    retrieved_k_results = retrieve_reference(index, lookup_indices, cite_rep, config, top_k=config["sc_retriever_topk"], silent=silent)

    return retrieved_k_results


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


def collect_retrieval_results(retrieved_k_results, retrieval_dataset, arxiv_to_corpus_id_map, sc_metadata_corpus, silent=False):
    references = []
    distances = []
    for each in retrieved_k_results:
        curr_corpus_idx, distance = each
        try:
            # sc corpus id -> arxiv id
            docs_corpus_idx = arxiv_to_corpus_id_map[sc_metadata_corpus[curr_corpus_idx]["paper_id"]]
        except KeyError:
            if not silent:
                print(f"arxiv id {sc_metadata_corpus[curr_corpus_idx]["paper_id"]} not found in the docs retrieval_dataset")
            continue
        references.append(retrieval_dataset[docs_corpus_idx])
        distances.append(distance)
    if len(references) == 0:
        raise ScholarCopilotRetrievalError(f"none of the retrieved results found in the retrieval_dataset")
    if not silent:
        print(f"best reference before passage retrieval: {references[0]["arxiv_id"]}")
    return references, distances


def replace_citations(current_text, unique_reference_id_list, retrieval_dataset, sc_metadata_corpus, docs_corpus_id_map, sc_corpus_id_map, silent=False):
    if not silent:
        print("IN replace_citations\n")
    # Find all citations with pattern <|cite_start|>XXX<|cite_end|>
    pattern = r'<\|cite_start\|>(.*?)<\|cite_end\|>'
    # Keep track of current citation index
    citation_index = 0
    new_citation_data = []
    last_replacement = ""

    # Function to replace each match with corresponding reference id
    def replace_match(match):
        nonlocal citation_index, new_citation_data, last_replacement
        if citation_index < len(unique_reference_id_list):
            
            arxiv_id = unique_reference_id_list[citation_index][0]
            unique_id_suffix = unique_reference_id_list[citation_index][1]

            try: # corpus id is from the docs dataset
                corpus_id = docs_corpus_id_map[arxiv_id]
                title = retrieval_dataset[corpus_id]["title"]
            except KeyError: # corpus id is from the sc corpus
                corpus_id = sc_corpus_id_map[arxiv_id]
                title = sc_metadata_corpus[corpus_id]["title"]

            citation_key = f"arxivID-{arxiv_id}-{unique_id_suffix}"
            # print("citation_key", citation_key)
            replacement = "\\cite{" + citation_key + "}"

            citation_data_entry = {
                "corpus_id": corpus_id,
                "arxiv_id": arxiv_id,
                "title": title,
                "citation_key": citation_key
            }
            # print("citation_data", citation_data)
            if last_replacement == replacement:
                replacement = ""
            else:
                if citation_index == len(unique_reference_id_list)-1: # the last entry is the new one
                    new_citation_data.append(citation_data_entry)
                last_replacement = replacement
            citation_index += 1
            return replacement
        return match.group(0), new_citation_data  # Keep original if no more reference ids

    # Replace all citations
    result = re.sub(pattern, replace_match, current_text)
    result = result.replace("<|paper_start|> ", "").replace("<|cite_start|>", "")
    # print("new_citation_data", new_citation_data)

    return result, new_citation_data


def post_process_output_text(res_text, unique_reference_id_list, retrieval_dataset, sc_metadata_corpus, docs_corpus_id_map, sc_corpus_id_map, silent=False):
    # print("post_process_output_text, res_text", res_text)
    output_text, citation_info_list = replace_citations(
        res_text, unique_reference_id_list, retrieval_dataset, sc_metadata_corpus, docs_corpus_id_map, sc_corpus_id_map, silent=silent
    )
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


def load_faiss_index(index_dir, lookup_indices_dir):
    index = faiss.read_index(index_dir)
    with open(lookup_indices_dir, 'rb') as f:
        lookup_indices = np.load(f, allow_pickle=True)
    return index, lookup_indices


def load_model(model_path, config):
    model_config = AutoConfig.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config)
    model.to(config["scholarcopilot_device"])

    tokenizer = AutoTokenizer.from_pretrained( # this is also used to apply the query context window for PR
        model_path,
        padding_side="left",
        truncation_side="left"
    )
    tokenizer.add_tokens(config["special_tokens"])

    model.resize_token_embeddings(len(tokenizer))
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print("scholarcopilot model loaded successfully")
    return model, tokenizer


class ScholarCopilotRetrievalError(Exception):
    """Scholar Copilot did retrieve less than one reference that can be found in the provided retrieval dataset."""
    pass

