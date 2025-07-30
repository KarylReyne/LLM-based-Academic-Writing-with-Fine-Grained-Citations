import json
import torch
import faiss
import sys
import ijson
import time
import random
import numpy as np

from evaluation_ranking_functions import rank_with_scholarcopilot, rank_with_scholarcopilot_with_passageretrieval
from passage_retrieval_interface import get_config, get_passage_retrieval_models, save_results
from scholarcopilot_model import load_model, load_faiss_index
from dataset_loaders import arxiv_to_corpus_id, load_retrieval_dataset, load_prebuilt_passages_dataset, load_eval_dataset
from util import recall_at_k


# Retrieval Accuracy. Citation retrieval is evaluated using Recall@k (k = 1 to 10), defined as
# the proportion of cases where the correct citation appears among the top-k retrieved results.
# Specifically, citations and subsequent content in 1,000 test samples are masked, and retrieval
# models predict citations based solely on the preceding context. For baseline models, we
# found that using the entire preceding context reduces performance; thus, we only use the
# last sentence before the citation as the query. Recall@k is computed by comparing predicted
# citations to the original ground-truth citations.
if __name__ == "__main__":
    RECALL_K = 10
    shuffled_samples = True

    config = get_config()

    model_path = "scholarcopilot_model_v1208/"
    model, tokenizer = load_model(model_path, config)

    passage_retrieval_models = get_passage_retrieval_models(config)

    id_map_path = "data/arxiv_to_corpus_id_documents_3.0.json"
    processed_corpus_path = "data/documents_3.0_processed_corpus.jsonl"
    arxiv_to_corpus_id_map = arxiv_to_corpus_id(id_map_path, processed_corpus_path)
    
    retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    complete_dataset_path = "data/documents_3.0_with_ids.jsonl"
    retrieval_dataset = load_retrieval_dataset(retrieval_dataset_path, complete_dataset_path, arxiv_to_corpus_id_map)

    index_dir = "data/index"
    lookup_indices_dir = "data/lookup_indices.npy"
    index, lookup_indices = load_faiss_index(index_dir, lookup_indices_dir)
    print("index building finished")
    
    # eval_dataset_path = f"data/context_citation_pairs_{config["query_context"]}_documents_3.0.jsonl"
    # eval_dataset_path = f"data/context_citation_pairs_last_sentence_documents_3.0.jsonl"
    eval_dataset_path = f"data/context_citation_pairs_intro+relwork_documents_3.0.jsonl"
    # eval_dataset_path = f"data/context_citation_pairs_intro+relwork_first1M_documents_3.0.jsonl"
    eval_dataset, eval_indices = load_eval_dataset(eval_dataset_path, arxiv_to_corpus_id_map, passage_retrieval_models["retr_tokenizer"], config, shuffle=shuffled_samples, max_samples=10000)

    if config["sc_retriever_topk"] < RECALL_K:
        raise ValueError(f"ScholarCopilot top-k ({config["sc_retriever_topk"]}) cannot be smaller than recall k ({RECALL_K})")
    if config["reranker_topk"] < RECALL_K:
        raise ValueError(f"Reranking top-k ({config["reranker_topk"]}) cannot be smaller than recall k ({RECALL_K})")
    
    sc_rankings = [] # len_dataset x recall_k
    sc_pr_rankings = [] # len_dataset x recall_k
    gold = [] # len_dataset x 1
    max_samples = 10
    samples = 0
    num_fails = 0
    print()
    for i in eval_indices:
        sys.stdout.write("\033[F")
        print(f"processing entry {samples}")

        item = eval_dataset[i]

        context = item["context"]
        gold.append(item["target_corpus_id"])

        debug_context = retrieval_dataset[item["target_corpus_id"]]["abstract"]+" <|cite_start|>" # should yield sim=1.0

        sc_rankings.append(rank_with_scholarcopilot(
            debug_context, index, lookup_indices, model, tokenizer, config
        ))

        ranking, fail = rank_with_scholarcopilot_with_passageretrieval(
            context, retrieval_dataset, arxiv_to_corpus_id_map, index, lookup_indices, model, tokenizer, passage_retrieval_models, config
        )
        sc_pr_rankings.append(ranking)
        num_fails += fail

        samples += 1
        if samples >= max_samples:
            break

    sc_recall = recall_at_k(sc_rankings, gold, k=RECALL_K)
    sc_pr_recall = recall_at_k(sc_pr_rankings, gold, k=RECALL_K)

    print(f"---recall@{RECALL_K} after {max_samples} retrievals---")
    print(f"ScholarCopilot: {sc_recall}")
    print(f"ScholarCopilot with passage retrieval: {sc_pr_recall} ({num_fails}/{samples} response failures)")

    save_results({
        "eval_dataset": eval_dataset_path,
        "num_samples": samples,
        "shuffled_samples": shuffled_samples,
        f"ScholarCopilot recall@{RECALL_K}": sc_recall,
        f"ScholarCopilot with passage retrieval recall@{RECALL_K}": sc_pr_recall,
        "passage retrieval fails": num_fails
    }, config, mode="eval_retrieval")
