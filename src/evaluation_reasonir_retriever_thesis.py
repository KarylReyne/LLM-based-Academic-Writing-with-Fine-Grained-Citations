import sys

from evaluation_ranking_functions import rank_with_scholarcopilot, rank_with_passage_retrieval_from_sc_rankings, rank_with_reasonir
from passage_retrieval_interface import get_config, get_passage_retrieval_models, save_results
from scholarcopilot_model import load_model, load_faiss_index, ScholarCopilotRetrievalError
from dataset_loaders import arxiv_to_corpus_id, load_retrieval_dataset, scholarcopilot_arxiv_to_corpus_id, load_sections_eval_dataset, load_scholarcopilot_metadata_corpus
from training_create_passage_encodings import load_passages_data_dataset
from util import recall_at_k, single_recall_at_k


if __name__ == "__main__":
    config = get_config("cfg/config_thesis_eval_reasonir_retrieval.json")

    model_path = "scholarcopilot_model_v1208/"
    model, tokenizer = load_model(model_path, config)

    passage_retrieval_models = get_passage_retrieval_models(config)

    docs_id_map_path = "data/arxiv_to_corpus_id_documents_3.0.json"
    processed_corpus_path = "data/documents_3.0_processed_corpus.jsonl"
    docs_corpus_id_map = arxiv_to_corpus_id(docs_id_map_path, processed_corpus_path)

    sc_corpus_id_map_path = "data/arxiv_to_corpus_id_scholar_copilot_train_data_500k.json"
    sc_corpus_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    sc_corpus_id_map = scholarcopilot_arxiv_to_corpus_id(sc_corpus_id_map_path, sc_corpus_path)
    sc_arxiv_id_map = {v: k for k, v in sc_corpus_id_map.items()}
    sc_to_docs_corpus_id = lambda x: docs_corpus_id_map[sc_arxiv_id_map[x]]
    
    docs_retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    complete_dataset_path = "data/documents_3.0_with_ids.jsonl"
    docs_retrieval_dataset = load_retrieval_dataset(docs_retrieval_dataset_path, complete_dataset_path, docs_corpus_id_map)

    corpus_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    sc_metadata_corpus = load_scholarcopilot_metadata_corpus(corpus_path)

    passages_data_dataset_path = "data/documents_3.0_512-passages_passages-data.jsonl"
    passages_data_dataset = load_passages_data_dataset(passages_data_dataset_path, docs_corpus_id_map)

    index_dir = "scholarcopilot_data/index"
    lookup_indices_dir = "scholarcopilot_data/lookup_indices.npy"
    index, lookup_indices = load_faiss_index(index_dir, lookup_indices_dir)
    reasonir_index_dir = "data/documents_3.0_512-passages_reasonir_8b-encoded_index"
    reasonir_lookup_indices_dir = "data/documents_3.0_512-passages_reasonir_8b-encoded_lookup_indices.npy"
    reasonir_index, reasonir_lookup_indices = load_faiss_index(reasonir_index_dir, reasonir_lookup_indices_dir)
    print("index building finished")

    SECTIONS = "intro+relwork"
    # SECTIONS = "methods"
    # SECTIONS = "experiments"
    # SECTIONS = "conclusion"

    with_abstracts = False
    shuffle = True

    if SECTIONS == "intro+relwork":
        target_sections = ["introduction", "related work"]
    elif SECTIONS == "methods":
        target_sections = ["methods"]
    elif SECTIONS == "experiments":
        target_sections = ["experiments"]
    elif SECTIONS == "conclusion":
        target_sections = ["conclusion"]
    insert = "_with_abstracts" if with_abstracts else ""
    eval_dataset_path = f"data_thesis/eval_dataset_{SECTIONS}-sections_documents_3.0_for_sc_corpus{insert}.jsonl"
    docs_dataset_path = "data/documents_3.0_with_ids.jsonl"
    eval_dataset, eval_indices = load_sections_eval_dataset(
        target_sections, 
        eval_dataset_path, 
        docs_dataset_path, 
        docs_retrieval_dataset, 
        docs_corpus_id_map, 
        sc_corpus_id_map, 
        config, 
        max_samples=100000, 
        populate_with_abstracts=with_abstracts, 
        shuffle=shuffle
    )

    RECALL_K = 10
    
    reasonir_rankings = []
    reasonir_pr_rankings = []
    gold = [] # len_dataset x 1
    max_samples = 1000
    samples = 0
    num_llm_fails = 0
    retrieval_fails = 0
    overlap_of_successful_retrievals = 0
    pr_fail_records = {}
    llm_response_fail_records = {}
    eps = 1e-6 # fail metrics

    # override some model settings to match the settings defined in this file
    config["only_passage_retriever_topk"] = RECALL_K*2
    config["reranker_topk"] = RECALL_K
    insert = "out" if not with_abstracts else ""
    config["custom_save_dir"] = f"out_thesis/eval_reasonir_retrieval_recall@{RECALL_K}_{max_samples}_{SECTIONS}/with{insert}_abstracts/"

    print()
    for i in eval_indices:
        sys.stdout.write("\033[F")
        print(f"processing entry {samples}")

        try:
            item = eval_dataset[i]
            context = item["context"]
            target_docs_corpus_id = docs_corpus_id_map[item["target_arxiv_id"]]

            reasonir_ranking, reasonir_references = rank_with_reasonir(
                context, passage_retrieval_models, reasonir_index, reasonir_lookup_indices, passages_data_dataset, docs_corpus_id_map, config
            )

            reasonir_pr_ranking, fail, reasonir_reranking_results, error_msg = rank_with_passage_retrieval_from_sc_rankings(
                context, reasonir_references, docs_corpus_id_map, tokenizer, passage_retrieval_models, config
            )

            # appending to the containers is delayed until all retrievals are done (because of the error handling)
            gold.append(target_docs_corpus_id)
            reasonir_rankings.append(reasonir_ranking)
            reasonir_pr_rankings.append(reasonir_pr_ranking)
            num_llm_fails += fail

            # determine overlap
            bool_sc = single_recall_at_k(reasonir_ranking, target_docs_corpus_id, RECALL_K)
            bool_sc_pr = single_recall_at_k(reasonir_pr_ranking, target_docs_corpus_id, RECALL_K)
            overlap_of_successful_retrievals += bool_sc and bool_sc_pr

            # collect pr failures
            if bool_sc and not bool_sc_pr:
                rec = {
                    "sample context": context,
                    "sample source": item["source_arxiv_id"],
                    "sample target": item["target_arxiv_id"],
                    "ranked_passage_labels": reasonir_reranking_results["ranked_passage_labels"], 
                    "ranked_passage_scores": reasonir_reranking_results["ranked_passage_scores"], 
                    "ranked_passages": reasonir_reranking_results["ranked_passages"]
                }
                pr_fail_records[f"sample index {i}"] = rec

            # collect failing llm responses
            if error_msg != None:
                rec = {
                    "sample context": context,
                    "sample source": item["source_arxiv_id"],
                    "sample target": item["target_arxiv_id"],
                    "failing llm chat log": error_msg
                }
                llm_response_fail_records[f"sample index {i}"] = rec

            samples += 1
            if samples >= max_samples:
                break

        except ScholarCopilotRetrievalError : 
            retrieval_fails += 1 # skip this sample entirely

    reasonir_recall = recall_at_k(reasonir_rankings, gold, k=RECALL_K)
    reasonir_pr_recall = recall_at_k(reasonir_pr_rankings, gold, k=RECALL_K)

    print(f"---recall@{RECALL_K} after {max_samples} retrievals---")
    print(f"ReasonIR passage retrieval: {reasonir_recall}")
    print(f"ReasonIR passage retrieval with reranking: {reasonir_pr_recall} ({num_llm_fails}/{samples} response failures)")

    save_results({
        "eval_dataset": eval_dataset_path,
        "eval_indices": [int(i) for i in eval_indices],
        "retrieval_index": index_dir,
        "num_samples": samples,
        "shuffled_samples": shuffle,
        f"ReasonIR passage retrieval recall@{RECALL_K}": reasonir_recall,
        f"ReasonIR passage retrieval with reranking recall@{RECALL_K}": reasonir_pr_recall,
        "overlap ratio of successful retrievals": (overlap_of_successful_retrievals/samples)/(reasonir_pr_recall+eps),
        "unsuccessful rerankings given successful SC retrieval": (len(pr_fail_records)/samples)/(reasonir_recall+eps),
        "llm response fails": num_llm_fails,
        "SC retrieval fails": retrieval_fails,
        "samples_where_only_pr_fails": pr_fail_records,
        "llm response failure logs": llm_response_fail_records
    }, config, mode="eval_retrieval")
