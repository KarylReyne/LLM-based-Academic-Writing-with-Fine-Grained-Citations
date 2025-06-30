import torch
import json
import os
import sys
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch import nn
import numpy as np
import itertools
from datetime import datetime
import time

from latex_parsing import *
from passage_retrieval import retrieval
from passage_reranking import reranking_and_scoring, InvalidLLMResponseError
from passage_retrieval_instructions import *


def get_config():
    config = None
    with open('cfg/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    assert config != None
    return config


def save_results(
    evaluation_records,
    config,
    mode="passage retrieval" # "passage retrieval" for saving passages, "generation" for saving generation output
):
    results = None
    try:
        with open(f'out/{datetime.now().strftime('%Y-%m-%d')}/evaluation_records.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        if not os.path.exists(f'out/{datetime.now().strftime('%Y-%m-%d')}'):
            os.makedirs(f'out/{datetime.now().strftime('%Y-%m-%d')}')
        results = {}
    assert os.path.exists(f'out/{datetime.now().strftime('%Y-%m-%d')}')
    assert results != None

    if mode == "passage retrieval":
        results["last changed"] = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        results["config"] = config
        results["instructions"] = {
            "retr_query": retrieval_instruction_query,
            "retr_document": retrieval_instruction_document,
            "rera_scoring": reranking_instruction("", [])
        }
        for key in evaluation_records:
            try:
                results["records"][key] = evaluation_records[key]
            except KeyError:
                results["records"] = {}
                results["records"][key] = evaluation_records[key]

    elif mode == "generation":
        results["last changed"] = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        for key in evaluation_records:
            results[key] = evaluation_records[key]

    else:
        raise NotImplementedError(f"mode '{mode}' is not implemented!")


    with open(f'out/{datetime.now().strftime('%Y-%m-%d')}/evaluation_records.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def get_passage_retrieval_models(config):
    # retrieval model definition
    retr_tokenizer = AutoTokenizer.from_pretrained(config["retriever"])
    retriever = AutoModel.from_pretrained(
        config["retriever"], 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    retriever = retriever.to(config["retriever_device"])
    retriever.eval()

    # reranker model definition
    rera_tokenizer = AutoTokenizer.from_pretrained(config["reranker"])
    reranker = AutoModelForCausalLM.from_pretrained(
        config["reranker"], 
        torch_dtype="auto",
        trust_remote_code=True
    )
    reranker = reranker.to(config["reranker_device"])
    reranker.eval()
    return {
        "retr_tokenizer": retr_tokenizer, 
        "retriever": retriever, 
        "rera_tokenizer": rera_tokenizer, 
        "reranker": reranker
    }


def get_candidate_passages(target_ids, tokenizer, config):
    target_doc_lines = [] # num_ids x num_lines
    target_doc_ids = [] # num_ids x num_lines
    for id in target_ids:
        try:
            _, lines = extract_full_latex_textbody(id)
        except AssertionError: # download if target not found
            download_from_arxiv(id)
            _, lines = extract_full_latex_textbody(id)
        for line in lines:
            target_doc_lines.append(line)
            target_doc_ids.append(id)
    target_doc_sections = get_target_sections(target_doc_lines, target_doc_ids, tokenizer, config)
    return target_doc_sections


def unified_passage_retrieval(generated_context, reference_ids, passage_retrieval_models, config):
    evaluation_records = {}
    evaluation_records[f"reference_ids-{reference_ids}"] = {
        "generated context": generated_context,
        "retrieved documents": {},
        "reranked documents": {},
        "final ranking": {}
    }

    candidate_passages = get_candidate_passages(
        reference_ids, passage_retrieval_models["retr_tokenizer"], config,
    )

    # --- RETRIEVAL ---
    # separate section labels and documents
    document_labels = []
    documents = []
    for d in candidate_passages:
        split = d.split(LABEL_SEPARATOR)
        document_labels.append(split[0])
        documents.append(split[1])

    retrieval(
        evaluation_records,
        generated_context, 
        document_labels, 
        documents,
        reference_ids,
        passage_retrieval_models["retriever"], 
        config
    )

    # --- RERANKING ---
    document_labels = []
    documents = []

    for doc_label, doc_dict in evaluation_records[f"reference_ids-{reference_ids}"]["retrieved documents"].items():
        document_labels.append(doc_label)
        documents.append(doc_dict["section chunk"])

    best_matching_passage, best_passage_label, best_passage_score, final_scores = reranking_and_scoring(
        evaluation_records, 
        generated_context,
        document_labels, 
        documents,
        reference_ids,
        passage_retrieval_models["reranker"], 
        passage_retrieval_models["rera_tokenizer"], 
        config
    )

    save_results(
        evaluation_records,
        config
    )

    return best_matching_passage, best_passage_label, best_passage_score, final_scores


def apply_retrieval_context_window(generated_context, tokenizer, config):
    tokens = tokenizer(generated_context).to("cuda:2")
    tokens = tokens["input_ids"] # get only the encoded tokens
    index = len(tokens)-1 # index of the citation, for generation always the last index
    low = max(index-config["query_context"], 0)
    high = index+1
    context = tokenizer.decode(tokens[low:high])
    context = context.replace(TOKENIZER_BEGIN_TOKEN, "")
    return context


def retrieve_relevant_passages(generated_context, reference_ids, passage_retrieval_models, config):
    if config["enable_query_context_window"]:
        generated_context = apply_retrieval_context_window(
            generated_context, passage_retrieval_models["retr_tokenizer"], config
        )
    best_matching_passage, best_passage_label, best_passage_score, final_scores = unified_passage_retrieval(
        generated_context,
        reference_ids, 
        passage_retrieval_models,
        config
    )
    # delete extracted archive
    dont_keep_folder = True
    for id in reference_ids:
        if dont_keep_folder and os.path.isdir(f'./data/{id}/'):
            shutil.rmtree(f"./data/{id}/")
    return best_matching_passage, best_passage_label, best_passage_score, final_scores
