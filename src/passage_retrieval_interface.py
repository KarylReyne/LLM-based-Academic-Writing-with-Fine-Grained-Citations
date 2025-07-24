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
    mode="passage retrieval", # "passage retrieval", "generation", "eval_retrieval", "eval_generation"
    save_passage_records=True
):
    results = None
    try:
        with open(f'out/{datetime.now().strftime('%Y-%m-%d')}/records.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        if not os.path.exists(f'out/{datetime.now().strftime('%Y-%m-%d')}'):
            os.makedirs(f'out/{datetime.now().strftime('%Y-%m-%d')}')
        results = {}
    assert os.path.exists(f'out/{datetime.now().strftime('%Y-%m-%d')}')
    assert results != None

    if mode == "passage retrieval": # saves continuously at depth 2
        results["last changed"] = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        results["config"] = config
        results["instructions"] = {
            "retr_query": retrieval_instruction_query(config["citation_mask_token"]),
            "retr_document": retrieval_instruction_document,
            "rera_scoring": reranking_instruction("", [])
        }
        if save_passage_records:
            for key in evaluation_records:
                try:
                    results["passage_records"][key] = evaluation_records[key]
                except KeyError:
                    results["passage_records"] = {}
                    results["passage_records"][key] = evaluation_records[key]

    elif mode in ["generation", "eval_retrieval", "eval_generation"]: # saves once at depth 1
        results["last changed"] = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        results["config"] = config
        for key in evaluation_records:
            results[key] = evaluation_records[key]

    else:
        raise NotImplementedError(f"mode '{mode}' is not implemented!")


    with open(f'out/{datetime.now().strftime('%Y-%m-%d')}/records.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    if mode in ["generation", "eval_retrieval", "eval_generation"]:
        os.rename( # rename results file for final save
            f'out/{datetime.now().strftime('%Y-%m-%d')}/records.json',
            f'out/{datetime.now().strftime('%Y-%m-%d')}/records_{mode}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json'
        )


def get_passage_retrieval_models(config):
    special_tokens = ['<|paper_start|>', '<|paper_end|>', '<|cite_start|>', '<|cite_end|>', '<|reference_start|>',
                      '<|reference_end|>', config["label_sep_token"], config["citation_mask_token"]]

    # tokenizers
    retr_tokenizer = AutoTokenizer.from_pretrained(config["retriever"])
    rera_tokenizer = AutoTokenizer.from_pretrained(config["reranker"])
    for tokenizer in [retr_tokenizer, rera_tokenizer]:
        # tokenizer.padding_side = 'right'
        tokenizer.add_tokens(special_tokens)
        # if tokenizer.pad_token_id is None:
        #     tokenizer.pad_token_id = tokenizer.eos_token_id

    # retrieval model definition
    retriever = AutoModel.from_pretrained(
        config["retriever"], 
        # torch_dtype="auto", 
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16
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
        torch_dtype=torch.bfloat16
    )
    reranker = reranker.to(config["reranker_device"])
    reranker.resize_token_embeddings(len(rera_tokenizer))
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
        for section in rec["sections"]:
            section_label = f"{rec["arxiv_id"]}_{section["title"].lstrip(" ").replace(" ", "-")}"
            text = " ".join(section["sentences"])
            tokens = tokenizer(text).to(config["retriever_device"])
            tokens = tokens["input_ids"] # get only the encoded tokens

            passage_idx = 0
            for i in range(0, len(tokens), config["passage_length"]):
                passage_label = f"{section_label}-{passage_idx}"
                passage = tokenizer.decode(tokens[i:i+config["passage_length"]]).replace(config["tokenizer_begin_token"], "")
                candidate_passages.append(passage_label+config["label_sep_token"]+passage)
                passage_idx += 1

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
    b = "passages" in references[0] and not "sections" in references[0] # given prebuilt passages
    if b:
        candidate_passages = [p for r in references for p in r["passages"]]
    else:
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
    ranked_passages, ranked_passage_labels, ranked_passage_scores, final_scores = reranking_and_scoring(
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

    return ranked_passages, ranked_passage_labels, ranked_passage_scores, final_scores


def apply_retrieval_context_window(generated_context, tokenizer, config):
    context = generated_context
    if config["enable_query_context_window"]:
        tokens = tokenizer(generated_context).to(config["retriever_device"])
        tokens = tokens["input_ids"] # get only the encoded tokens
        index = len(tokens)-1 # index of the citation, for generation always the last index
        low = max(index-config["query_context"], 0)
        high = index+1
        context = tokenizer.decode(tokens[low:high])
        context = context.replace(config["tokenizer_begin_token"], "")
    return context


def retrieve_relevant_passages(generated_context, references, passage_retrieval_models, config, silent=False, save_passage_records=True):
    if config["enable_query_context_window"]: # for evaluation, this is redundant, see evaluation_ranking_functions.py
        generated_context = apply_retrieval_context_window(
            generated_context, passage_retrieval_models["retr_tokenizer"], config
        ) 
    ranked_passages, ranked_passage_labels, ranked_passage_scores, final_scores = unified_passage_retrieval(
        generated_context,
        references, 
        passage_retrieval_models,
        config,
        silent=silent,
        save_passage_records=save_passage_records
    )
    return ranked_passages, ranked_passage_labels, ranked_passage_scores, final_scores
