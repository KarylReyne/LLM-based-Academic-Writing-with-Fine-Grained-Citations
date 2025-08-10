import sys

from evaluation_ranking_functions import rank_with_scholarcopilot, rank_with_passage_retrieval_from_sc_rankings
from passage_retrieval_interface import get_config, get_passage_retrieval_models, save_results
from scholarcopilot_model import load_model, load_faiss_index, ScholarCopilotRetrievalError
from dataset_loaders import arxiv_to_corpus_id, load_retrieval_dataset, load_eval_dataset, scholarcopilot_arxiv_to_corpus_id, load_sections_eval_dataset, load_retrieval_dataset_for_sc_corpus_with_fulltext
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

    id_map_path = "data/arxiv_to_corpus_id_documents_3.0.json"
    processed_corpus_path = "data/documents_3.0_processed_corpus.jsonl"
    docs_id_map = arxiv_to_corpus_id(id_map_path, processed_corpus_path)

    sc_corpus_id_map = scholarcopilot_arxiv_to_corpus_id("data/arxiv_to_corpus_id_scholar_copilot_train_data_500k.json", "scholarcopilot_data/corpus_data_arxiv_1215.jsonl")
    sc_arxiv_id_map = {v: k for k, v in sc_corpus_id_map.items()}
    
    retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    complete_dataset_path = "data/documents_3.0_with_ids.jsonl"
    docs_retrieval_dataset = load_retrieval_dataset(retrieval_dataset_path, complete_dataset_path, docs_id_map)

    sc_retrieval_dataset_path = "data/retrieval_dataset_scholar_copilot_corpus_with_fulltext.jsonl"
    documents_retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    sc_retrieval_corpus_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    sc_retrieval_dataset = load_retrieval_dataset_for_sc_corpus_with_fulltext(
        sc_retrieval_dataset_path,
        {}, #docs_retrieval_dataset, # only needed when generating this dataset
        sc_retrieval_corpus_path,
        docs_id_map,
        config
    )

    index_dir = "data/index"
    lookup_indices_dir = "data/lookup_indices.npy"
    # index_dir = "scholarcopilot_data/index"
    # lookup_indices_dir = "scholarcopilot_data/lookup_indices.npy"
    index, lookup_indices = load_faiss_index(index_dir, lookup_indices_dir)
    print("index building finished")
    
    # eval_dataset_path = f"data/context_citation_pairs_256_documents_3.0.jsonl"
    # eval_dataset_path = f"data/context_citation_pairs_last_sentence_documents_3.0.jsonl"
    # eval_dataset_path = f"data/context_citation_pairs_intro+relwork_documents_3.0.jsonl"
    # eval_dataset, eval_indices = load_eval_dataset(eval_dataset_path, docs_id_map, passage_retrieval_models["retr_tokenizer"], config, shuffle=True, max_samples=10000)

    target_sections = ["methods"]
    sections_eval_dataset_path = "data/eval_dataset_methods-sections_documents_3.0_for_sc_corpus_with_abstracts.jsonl"
    docs_dataset_path = "data/documents_3.0_with_ids.jsonl"
    shuffle = True
    eval_dataset, eval_indices = load_sections_eval_dataset(target_sections, sections_eval_dataset_path, docs_dataset_path, docs_retrieval_dataset, docs_id_map, sc_corpus_id_map, max_samples=1000, populate_with_abstracts=True, shuffle=shuffle)

    RECALL_K = 5

    if config["sc_retriever_topk"] < RECALL_K:
        raise ValueError(f"ScholarCopilot top-k ({config["sc_retriever_topk"]}) cannot be smaller than recall k ({RECALL_K})")
    if config["reranker_topk"] < RECALL_K:
        raise ValueError(f"Reranking top-k ({config["reranker_topk"]}) cannot be smaller than recall k ({RECALL_K})")
    
    sc_rankings = [] # len_dataset x recall_k
    sc_pr_rankings = [] # len_dataset x recall_k
    gold = [] # len_dataset x 1
    max_samples = 10
    samples = 0
    num_llm_fails = 0
    retrieval_fails = 0
    overlap_of_successful_retrievals = 0
    pr_fail_records = {}
    llm_response_fail_records = {}
    eps = 1e-6 # fail metrics
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
            sc_pr_ranking, fail, reranking_results, error_msg = rank_with_passage_retrieval_from_sc_rankings(
                context, retrieved_k_results, docs_retrieval_dataset, sc_corpus_id_map, tokenizer, passage_retrieval_models, config
            )

            # appending to the containers is delayed until all retrievals are done (because of the error handling)
            gold.append(item["target_corpus_id"])
            sc_rankings.append(sc_ranking)
            sc_pr_rankings.append(sc_pr_ranking)
            num_llm_fails += fail

            # determine overlap
            bool_sc = single_recall_at_k(sc_ranking, item["target_corpus_id"], RECALL_K)
            bool_sc_pr = single_recall_at_k(sc_pr_ranking, item["target_corpus_id"], RECALL_K)
            overlap_of_successful_retrievals += bool_sc and bool_sc_pr

            # collect pr failures
            if bool_sc and not bool_sc_pr:
                rec = {
                    "sample context": context,
                    "sample source": item["source_arxiv_id"],
                    "sample target": sc_arxiv_id_map[item["target_corpus_id"]],
                    "ranked_passage_labels": reranking_results["ranked_passage_labels"], 
                    "ranked_passage_scores": reranking_results["ranked_passage_scores"], 
                    "ranked_passages": reranking_results["ranked_passages"]
                }
                pr_fail_records[f"sample index {i}"] = rec

            # collect failing llm responses
            if error_msg != None:
                rec = {
                    "sample context": context,
                    "sample source": item["source_arxiv_id"],
                    "sample target": sc_arxiv_id_map[item["target_corpus_id"]],
                    "failing llm chat log": error_msg
                }
                llm_response_fail_records[f"sample index {i}"] = rec

            samples += 1
            if samples >= max_samples:
                break

        except ScholarCopilotRetrievalError: 
            retrieval_fails += 1
            continue # skip this sample entirely

    print(retrieval_fails)
    sc_recall = recall_at_k(sc_rankings, gold, k=RECALL_K)
    sc_pr_recall = recall_at_k(sc_pr_rankings, gold, k=RECALL_K)

    print(f"---recall@{RECALL_K} after {max_samples} retrievals---")
    print(f"ScholarCopilot: {sc_recall}")
    print(f"ScholarCopilot with passage retrieval: {sc_pr_recall} ({num_llm_fails}/{samples} response failures)")

    save_results({
        "eval_dataset": sections_eval_dataset_path,
        "retrieval_index": index_dir,
        "num_samples": samples,
        "shuffled_samples": shuffle,
        f"ScholarCopilot recall@{RECALL_K}": sc_recall,
        f"ScholarCopilot with passage retrieval recall@{RECALL_K}": sc_pr_recall,
        "overlap ratio of successful retrievals": (overlap_of_successful_retrievals/samples)/(sc_pr_recall+eps),
        "unsuccessful rerankings given successful SC retrieval": (len(pr_fail_records)/samples)/(sc_recall+eps),
        "llm response fails": num_llm_fails,
        "SC retrieval fails": retrieval_fails,
        "samples_where_only_pr_fails": pr_fail_records,
        "llm response failure logs": llm_response_fail_records
    }, config, mode="eval_retrieval")
