import json
import sys
import itertools

from passage_retrieval_interface import get_config, get_passage_retrieval_models, retrieve_relevant_passages
from scholarcopilot_model import collect_retrieval_results
from scholarcopilot_generation import single_step_retrieval
from util import recall_at_k


def rank_with_scholarcopilot(context, recall_k, index, lookup_indices, model, tokenizer, config):
    retrieved_k_results = single_step_retrieval(context, recall_k, index, lookup_indices, model, tokenizer, config)
    ranked_corpus_ids = retrieved_k_results[:][0]
    return ranked_corpus_ids


def rank_with_scholarcopilot_with_passageretrieval(context, recall_k, retrieval_dataset, arxiv_to_corpus_id_map, index, lookup_indices, model, tokenizer, config):
    retrieved_k_results = single_step_retrieval(context, recall_k, index, lookup_indices, model, tokenizer, config)
    references, _ = collect_retrieval_results(retrieved_k_results, retrieval_dataset)
    ranked_passages, ranked_passage_labels, ranked_passage_scores, _ = retrieve_relevant_passages(
        context, references, passage_retrieval_models, config
    )
    if not len(ranked_passages) >= recall_k:
        raise ValueError(f"Reranking top-k ({len(ranked_passages)}) cannot be smaller than recall k ({recall_k})")
    ranked_corpus_ids = [arxiv_to_corpus_id_map[l.split("_")[0]] for l in ranked_passage_labels]
    return ranked_corpus_ids


if __name__ == "__main__":
    RECALL_K = 1

    config = get_config()

    model_path = "scholarcopilot_model_v1208/"
    model, tokenizer = load_model(model_path, config)

    passage_retrieval_models = get_passage_retrieval_models(config)
    print("models loaded")

    id_map_path = "data/arxiv_to_corpus_id_documents_3.0.json"
    processed_corpus_path = "data/documents_3.0_processed_corpus.jsonl"
    arxiv_to_corpus_id_map = arxiv_to_corpus_id(id_map_path, processed_corpus_path)
    
    retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    complete_dataset_path = "data/documents_3.0_with_ids.jsonl"
    retrieval_dataset = load_retrieval_dataset(retrieval_dataset_path, complete_dataset_path, arxiv_to_corpus_id_map)

    index_dir = "data/"
    index, lookup_indices = load_faiss_index(index_dir)
    print("index building finished")

    eval_dataset_path = f"data/context_citation_pairs_{config["query_context"]}_documents_3.0.jsonl"
    eval_dataset = {}
    with open(eval_dataset_path, "r") as file:
        eval_dataset = json.load(file)
    print("dataset loaded")

    
    sc_rankings = [] # len_dataset x recall_k
    sc_pr_rankings = [] # len_dataset x recall_k
    gold = [] # len_dataset x 1
    max_samples = 1000
    samples = 0
    for d in eval_dataset:
        context = d["context"]
        gold.append(d["target_corpus_id"])

        sc_rankings.append(rank_with_scholarcopilot(
            context, recall_k, index, lookup_indices, model, tokenizer, config
        ))
        sc_pr_rankings.append(rank_with_scholarcopilot_with_passageretrieval(
            context, recall_k, retrieval_dataset, arxiv_to_corpus_id_map, index, lookup_indices, model, tokenizer, config
        ))
        samples += 1
        if samples >= max_samples:
            break

    sc_recall = recall_at_k(sc_rankings, gold, k=RECALL_K)
    sc_pr_recall = recall_at_k(sc_pr_rankings, gold, k=RECALL_K)

    print(f"---recall@{RECALL_K} after {max_samples} retrievals---")
    print(f"ScholarCopilot: {sc_recall}")
    print(f"ScholarCopilot with passage retrieval: {sc_pr_recall}")
