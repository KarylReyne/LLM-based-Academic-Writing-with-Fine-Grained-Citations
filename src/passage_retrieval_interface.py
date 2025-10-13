import torch
import json
import os
import sys
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch import nn
import numpy as np
import faiss
from datetime import datetime
import time

from passage_retrieval import retrieval
from passage_reranking import reranking_and_scoring, InvalidLLMResponseError
from passage_retrieval_instructions import *
from evaluation_generation_instruction import *


def get_config(path='cfg/config.json'):
    config = None
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    assert config != None
    return config


def save_results(
    evaluation_records,
    config,
    mode="passage retrieval", # "passage retrieval", "generation", "eval_retrieval", "eval_generation"
    save_passage_records=True
):
    out_dir = "out"
    if "custom_save_dir" in config:
        out_dir = config["custom_save_dir"]
    results = None
    date_str = datetime.now().strftime('%Y-%m-%d')
    try:
        with open(f'{out_dir}/{date_str}/records.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        if not os.path.exists(f'{out_dir}/{date_str}'):
            os.makedirs(f'{out_dir}/{date_str}')
        results = {}
    assert os.path.exists(f'{out_dir}/{date_str}')
    assert results != None

    results["last changed"] = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    results["config"] = config
    results["instructions"] = {
        "retr_query": retrieval_instruction_query,
        "retr_document": retrieval_instruction_document,
        "rera_scoring": reranking_instruction("", []),
        "judge_instruction": judge_instruction2("", "", "", "")
    }

    if mode == "passage retrieval": # saves continuously at depth 2
        if save_passage_records:
            for key in evaluation_records:
                try:
                    results["passage_records"][key] = evaluation_records[key]
                except KeyError:
                    results["passage_records"] = {}
                    results["passage_records"][key] = evaluation_records[key]

    elif mode in ["generation", "eval_retrieval", "eval_generation"]: # saves once at depth 1
        for key in evaluation_records:
            results[key] = evaluation_records[key]

    else:
        raise NotImplementedError(f"mode '{mode}' is not implemented!")


    with open(f'{out_dir}/{date_str}/records.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    if mode in ["generation", "eval_retrieval", "eval_generation"]:
        date_exact = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.rename( # rename results file for final save
            f'{out_dir}/{date_str}/records.json',
            f'{out_dir}/{date_str}/records_{mode}_{date_exact}.json'
        )


def get_passage_retrieval_models(config):
    # tokenizers
    retr_tokenizer = AutoTokenizer.from_pretrained(
        config["retriever"]
    )
    rera_tokenizer = AutoTokenizer.from_pretrained(
        config["reranker"]
    )
    for tokenizer in [retr_tokenizer, rera_tokenizer]:
        tokenizer.add_tokens(config["special_tokens"])

    # retrieval model definition
    retriever = AutoModel.from_pretrained(
        config["retriever"], 
        # torch_dtype="auto", 
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
        torch_dtype=torch.float16
    )
    retriever = retriever.to(config["retriever_device"])
    retriever.resize_token_embeddings(len(retr_tokenizer))
    retriever.eval()

    # reranker model definition
    reranker = AutoModelForCausalLM.from_pretrained(
        config["reranker"], 
        # torch_dtype="auto",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
        torch_dtype=torch.float16
    )
    reranker = reranker.to(config["reranker_device"])
    reranker.resize_token_embeddings(len(rera_tokenizer))
    reranker.generation_config.pad_token_id = rera_tokenizer.pad_token_id
    reranker.eval()

    print("passage retrieval models loaded")
    return {
        "retr_tokenizer": retr_tokenizer, 
        "retriever": retriever, 
        "rera_tokenizer": rera_tokenizer, 
        "reranker": reranker
    }


def get_candidate_passages(references, tokenizer, config):
    candidate_passages = []
    for rec in references:
        if "sections" in rec: # rec is from a retrieval dataset
            for section in rec["sections"]:
                arxiv_id = rec["arxiv_id"]
                title = section["title"].lstrip(" ").replace(" ", "-")
                section_label = f"{arxiv_id}_{title}"
                text = " ".join(section["sentences"])
                tokens = tokenizer(text).to(config["retriever_device"])
                tokens = tokens["input_ids"] # get only the encoded tokens

                passage_idx = 0
                for i in range(0, len(tokens), config["passage_length"]):
                    passage_label = f"{section_label}-{passage_idx}"
                    passage = tokenizer.decode(tokens[i:i+config["passage_length"]]).replace(config["tokenizer_begin_token"], "")
                    candidate_passages.append(passage_label+config["label_sep_token"]+passage)
                    passage_idx += 1
        elif "passage_label" in rec: # rec is from retrieve_passages_with_reasonir()
            passage_label = rec["passage_label"]
            passage = rec["passage"]
            candidate_passages.append(passage_label+config["label_sep_token"]+passage)
        else:
            print(f"references: {references}")
            raise ValueError(f"get_candidate_passages() could not parse the given references.")

    return candidate_passages


def unified_passage_retrieval(generated_context, references, passage_retrieval_models, config, silent=False, save_passage_records=True):
    evaluation_records = {}
    reference_ids = [d["arxiv_id"] for d in references]
    evaluation_records[f"reference_ids-{reference_ids}"] = {
        "generated context": generated_context,
        "retrieved documents": {},
        "reranked documents": {},
        "final ranking": {}
    }

    start = time.time()
    candidate_passages = get_candidate_passages(
        references, passage_retrieval_models["retr_tokenizer"], config
    )
    # print("***************Build passages cost (time): ", time.time() - start)

    start = time.time()
    # --- RETRIEVAL ---
    # separate section labels and documents
    document_labels = []
    documents = []
    for d in candidate_passages:
        split = d.split(config["label_sep_token"])
        document_labels.append(split[0])
        documents.append(split[1])

    if config["enable_passage_retriever"]:
        retrieval(
            evaluation_records,
            generated_context, 
            document_labels, 
            documents,
            reference_ids,
            passage_retrieval_models["retriever"],
            passage_retrieval_models["retr_tokenizer"],
            config,
            silent=silent
        )
        
        document_labels = []
        documents = []
        for doc_label, doc_dict in evaluation_records[f"reference_ids-{reference_ids}"]["retrieved documents"].items():
            document_labels.append(doc_label)
            documents.append(doc_dict["section chunk"])
    # print("***************Retrieval cost (time): ", time.time() - start)

    start = time.time()
    # --- RERANKING ---
    reranking_results = reranking_and_scoring(
        evaluation_records, 
        generated_context,
        document_labels, 
        documents,
        reference_ids,
        passage_retrieval_models["reranker"], 
        passage_retrieval_models["rera_tokenizer"], 
        config,
        silent=silent
    )
    # print("***************Reranking cost (time): ", time.time() - start)

    save_results(
        evaluation_records,
        config,
        mode="passage retrieval",
        save_passage_records=save_passage_records
    )

    return reranking_results


def retrieve_passages_with_reasonir(index, lookup_indices, cite_start_hidden_state, passages_data_dataset, config, top_k=5, silent=False):
    start = time.time()
    if not silent:
        print("[ReasonIR-passages] Retrieving passages...")

    if isinstance(cite_start_hidden_state, torch.Tensor):
        cite_start_hidden_state = cite_start_hidden_state.cpu().numpy()

    if cite_start_hidden_state.ndim == 1: # transposes the vector
        cite_start_hidden_state = cite_start_hidden_state.reshape(1, -1)

    faiss.normalize_L2(cite_start_hidden_state)

    # custom efSearch
    index.hnsw.efSearch = config["hnsw_efSearch"]

    # cpu index search
    distances, indices = index.search(cite_start_hidden_state, top_k)

    retrieved_passage_labels = []
    for i in indices[0]:
        try:
            each_index = str(lookup_indices[i], 'ascii')
        except UnicodeDecodeError as e:
            each_index = str(lookup_indices[i], 'utf-8')
        retrieved_passage_labels.append(each_index)
    if not silent:
        print("[ReasonIR-passages] retrieved_passage_labels", retrieved_passage_labels)
        print("[ReasonIR-passages] distances[0]", distances[0])
        print("[ReasonIR-passages] ***************Retrieval cost (time): ", time.time() - start)

    references = []
    for label in retrieved_passage_labels:
        rec = passages_data_dataset[label]
        references.append(rec)
    
    return references, distances[0]


def apply_retrieval_context_window(generated_context, tokenizer, config):
    context = generated_context
    if config["enable_query_context_window"]:
        assert tokenizer.padding_side == "left"
        assert tokenizer.truncation_side == "left"
        tokens = tokenizer(
            generated_context,
            max_length=config["query_context"],
            padding="max_length",
            truncation=True
        ).to(config["retriever_device"])
        w = config["query_context"]
        assert len(tokens.input_ids) == w, f"{len(tokens.input_ids)} != {w}"
        context = tokenizer.decode(tokens.input_ids)
        context = context.replace(config["tokenizer_begin_token"], "")
    return context


def retrieve_relevant_passages(generated_context, references, tokenizer, passage_retrieval_models, config, silent=False, save_passage_records=True):
    if config["enable_query_context_window"]:
        generated_context = apply_retrieval_context_window(generated_context, tokenizer, config)
    reranking_results = unified_passage_retrieval(
        generated_context,
        references, 
        passage_retrieval_models,
        config,
        silent=silent,
        save_passage_records=save_passage_records
    )
    return reranking_results
