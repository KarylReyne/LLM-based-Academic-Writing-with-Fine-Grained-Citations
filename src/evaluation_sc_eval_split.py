import json
import torch
import faiss
import sys
import ijson
import time
import random
import ast
import numpy as np

from evaluation_ranking_functions import rank_with_scholarcopilot, rank_with_passage_retrieval_from_sc_rankings
from passage_retrieval_interface import get_config, get_passage_retrieval_models, save_results
from scholarcopilot_model import load_model, load_faiss_index, ScholarCopilotRetrievalError
from dataset_loaders import arxiv_to_corpus_id, scholarcopilot_arxiv_to_corpus_id, load_retrieval_dataset_from_sc_eval, load_retrieval_dataset_for_sc_corpus_with_fulltext, load_retrieval_dataset, load_scholarcopilot_eval_dataset
from util import recall_at_k, single_recall_at_k


# Retrieval Accuracy. Citation retrieval is evaluated using Recall@k (k = 1 to 10), defined as
# the proportion of cases where the correct citation appears among the top-k retrieved results.
# Specifically, citations and subsequent content in 1,000 test samples are masked, and retrieval
# models predict citations based solely on the preceding context. For baseline models, we
# found that using the entire preceding context reduces performance; thus, we only use the
# last sentence before the citation as the query. Recall@k is computed by comparing predicted
# citations to the original ground-truth citations.
if __name__ == "__main__":
    config = get_config()

    model_path = "scholarcopilot_model_v1208/"
    model, tokenizer = load_model(model_path, config)

    passage_retrieval_models = get_passage_retrieval_models(config)

    sc_corpus_id_map_path = "data/arxiv_to_corpus_id_scholar_copilot_train_data_500k.json"
    sc_corpus_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    sc_corpus_id_map = scholarcopilot_arxiv_to_corpus_id(sc_corpus_id_map_path, sc_corpus_path)

    documents_id_map_path = "data/arxiv_to_corpus_id_documents_3.0.json"
    processed_corpus_path = "data/documents_3.0_processed_corpus.jsonl"
    documents_arxiv_to_corpus_id_map = arxiv_to_corpus_id(documents_id_map_path, processed_corpus_path)
    
    # retrieval_dataset_path = "data/retrieval_dataset_scholar_copilot_train_data_500k.jsonl"
    # complete_dataset_path = "data_train/scholar_copilot_train_data_500k.json"
    # retrieval_dataset = load_retrieval_dataset_from_sc_eval(retrieval_dataset_path, complete_dataset_path, sc_corpus_id_map, config)

    # documents_retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    # complete_dataset_path = "data/documents_3.0_with_ids.jsonl"
    # documents_retrieval_dataset = load_retrieval_dataset(documents_retrieval_dataset_path, complete_dataset_path, documents_arxiv_to_corpus_id_map)

    index_dir = "scholarcopilot_data/index"
    lookup_indices_dir = "scholarcopilot_data/lookup_indices.npy"
    index, lookup_indices = load_faiss_index(index_dir, lookup_indices_dir)
    print("index building finished")

    retrieval_dataset_path = "data/retrieval_dataset_scholar_copilot_corpus_with_fulltext.jsonl"
    documents_retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    sc_retrieval_corpus_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    retrieval_dataset = load_retrieval_dataset_for_sc_corpus_with_fulltext(
        retrieval_dataset_path,
        {}, #documents_retrieval_dataset, # only needed when generating this dataset
        sc_retrieval_corpus_path,
        documents_arxiv_to_corpus_id_map,
        config
    )

    shuffle = True
    sc_eval_dataset_path = f"data_train/scholar_copilot_eval_data_1k.json"
    eval_dataset_path = sc_eval_dataset_path.replace(".json", "_eval_pairs.jsonl")
    eval_dataset, eval_indices = load_scholarcopilot_eval_dataset(eval_dataset_path, sc_eval_dataset_path, config, shuffle=shuffle)

    RECALL_K = 5

    if config["sc_retriever_topk"] < RECALL_K:
        raise ValueError(f"ScholarCopilot top-k ({config["sc_retriever_topk"]}) cannot be smaller than recall k ({RECALL_K})")
    
    sc_rankings = [] # len_dataset x recall_k
    sc_pr_rankings = [] # len_dataset x recall_k
    gold = [] # len_dataset x 1
    max_samples = 100
    samples = 0
    num_llm_fails = 0
    retrieval_fails = 0
    overlap_of_successful_retrievals = 0
    pr_fail_records = {}
    print()
    for i in eval_indices:
        sys.stdout.write("\033[F")
        print(f"processing entry {samples}")

        try:
            item = eval_dataset[i]
            context = item["context"]

            sc_ranking, retrieved_k_results = rank_with_scholarcopilot(
                context, index, lookup_indices, model, tokenizer, config
            )
            sc_pr_ranking, fail, ranked_passages, ranked_passage_labels, ranked_passage_scores = rank_with_passage_retrieval_from_sc_rankings(
                context, retrieved_k_results, retrieval_dataset, sc_corpus_id_map, tokenizer, passage_retrieval_models, config
            )

            # appending to the containers is delayed until all retrievals are done (because of the error handling)
            gold.append(item["citation_corpus_id"])
            sc_rankings.append(sc_ranking)
            sc_pr_rankings.append(sc_pr_ranking)
            num_llm_fails += fail

            # determine overlap
            bool_sc = single_recall_at_k(sc_ranking, item["citation_corpus_id"], RECALL_K)
            bool_sc_pr = single_recall_at_k(sc_pr_ranking, item["citation_corpus_id"], RECALL_K)
            overlap_of_successful_retrievals += bool_sc and bool_sc_pr

            # collect pr failures
            if bool_sc and not bool_sc_pr:
                rec = {
                    "sample context": context,
                    "sample target": item["citation_corpus_id"],
                    "ranked_passage_labels": ranked_passage_labels, 
                    "ranked_passage_scores": ranked_passage_scores, 
                    "ranked_passages": ranked_passages
                }
                pr_fail_records[f"sample index {i}"] = rec

            samples += 1
            if samples >= max_samples:
                break

        except ScholarCopilotRetrievalError: 
            retrieval_fails += 1
            continue # skip this sample entirely


    sc_recall = recall_at_k(sc_rankings, gold, k=RECALL_K)
    sc_pr_recall = recall_at_k(sc_pr_rankings, gold, k=RECALL_K)

    print(f"---recall@{RECALL_K} after {max_samples} retrievals---")
    print(f"ScholarCopilot: {sc_recall}")
    print(f"ScholarCopilot with passage retrieval: {sc_pr_recall} ({num_llm_fails}/{samples} response failures)")

    save_results({
        "eval_dataset": eval_dataset_path,
        "num_samples": samples,
        "shuffled_samples": shuffle,
        f"ScholarCopilot recall@{RECALL_K}": sc_recall,
        f"ScholarCopilot with passage retrieval recall@{RECALL_K}": sc_pr_recall,
        "overlap ratio of successful retrievals": overlap_of_successful_retrievals/samples,
        "llm response fails": num_llm_fails,
        "SC retrieval fails": retrieval_fails,
        "samples_where_only_pr_fails": pr_fail_records
    }, config, mode="eval_retrieval")
