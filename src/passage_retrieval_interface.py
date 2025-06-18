import torch
import json
import os
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch import nn
import numpy as np
import itertools
from datetime import datetime

from latex_parsing import download_from_arxiv, search_arxiv_for_citations_data, get_source_citations, LABEL_SEPARATOR, CITATION_MASK
from passage_retrieval import retrieval
from passage_reranking import reranking_and_scoring
from passage_retrieval_instructions import *


def get_config():
    config = None
    with open('cfg/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    assert config != None
    return config


def save_results(
    evaluation_records,
    config
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

    results[f"{datetime.now().strftime('%H-%M-%S')}"] = {
        "config": config,
        "instructions": {
            "retr_query": retrieval_instruction_query,
            "retr_document": retrieval_instruction_document,
            "rera_scoring": reranking_instruction("", [])
        },
        "records": evaluation_records
    }

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
    retriever = retriever.to("cuda")
    retriever.eval()

    # reranker model definition
    rera_tokenizer = AutoTokenizer.from_pretrained(config["reranker"])
    reranker = AutoModelForCausalLM.from_pretrained(
        config["reranker"], 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    reranker = reranker.to("cuda")
    return retr_tokenizer, retriever, rera_tokenizer, reranker


def get_candidate_passages(target_id, tokenizer, config):
    target_doc_lines = None
    try:
        _, target_doc_lines = extract_full_latex_textbody(target_id)
    except AssertionError: # download if target not found
        download_from_arxiv(target_id)
        _, target_doc_lines = extract_full_latex_textbody(target_id)
    assert target_doc_lines != None
    target_doc_sections = get_target_sections(target_doc_lines, tokenizer, with_chunking=config["target_chunking"], chunk_size=config["chunk_size"])
    return target_doc_sections


def unified_passage_retrieval(generated_context, target_id, config):
    retr_tokenizer, retriever, rera_tokenizer, reranker = get_passage_retrieval_models(config)

    target_doc_sections = get_candidate_passages(target_id, retr_tokenizer, config)

    evaluation_records = {}
    evaluation_records["ScholarCopilot_Generation"] = {
        "generated context": generated_context,
        "retrieved documents": {},
        "reranked documents": {},
        "final ranking": {}
    }

    # --- RETRIEVAL ---
    # separate section labels and documents
    document_labels = []
    documents = []
    for d in target_doc_sections:
        split = d.split(LABEL_SEPARATOR)
        document_labels.append(split[0])
        documents.append(split[1])

    retrieval(
        evaluation_records,
        generated_context, 
        document_labels, 
        documents,
        retriever, 
        config
    )

    # --- RERANKING ---
    document_labels = []
    documents = []

    for doc_label, doc_dict in evaluation_records["ScholarCopilot_Generation"]["retrieved documents"].items():
        document_labels.append(doc_label)
        documents.append(doc_dict["section chunk"])

    assert len(documents) == config["retriever_topk"], f"{len(documents)}, {config["retriever_topk"]}"

    reranking_and_scoring(
        evaluation_records, 
        generated_context,
        document_labels, 
        documents,
        reranker, 
        rera_tokenizer, 
        config
    )

    save_results(
        evaluation_records,
        config
    )


def apply_retrieval_context_window(generated_context, config):
    pass


def retrieve_relevant_passages(generated_context, target_id, num_passages, tokenizer):
    config = get_config()
    generated_context = apply_retrieval_context_window(generated_context)
    unified_passage_retrieval(generated_context, candidate_passages, config)

