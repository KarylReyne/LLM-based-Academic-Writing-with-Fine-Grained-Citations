import json
import torch
import faiss
import sys
import ijson
import time
import random
import ast
import numpy as np

from evaluation_ranking_functions import rank_with_scholarcopilot, rank_with_scholarcopilot_with_passageretrieval
from passage_retrieval_interface import get_config, get_passage_retrieval_models
from scholarcopilot_model import load_model, load_faiss_index
from dataset_loaders import scholarcopilot_arxiv_to_corpus_id, load_retrieval_dataset_from_sc_eval
from util import recall_at_k


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
    
    retrieval_dataset_path = "data/retrieval_dataset_scholar_copilot_train_data_500k.jsonl"
    complete_dataset_path = "data_train/scholar_copilot_train_data_500k.json"
    retrieval_dataset = load_retrieval_dataset_from_sc_eval(retrieval_dataset_path, complete_dataset_path, sc_corpus_id_map, config)

    index_dir = "scholarcopilot_data/index"
    lookup_indices_dir = "scholarcopilot_data/lookup_indices.npy"
    index, lookup_indices = load_faiss_index(index_dir, lookup_indices_dir)
    print("index building finished")

    eval_dataset_path = f"data_train/scholar_copilot_eval_data_1k.json"
    out_file = eval_dataset_path.replace(".json", "_eval_pairs.jsonl")
    eval_dataset = []
    count = 0
    print()
    try:
        with open(out_file, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {count}")
                eval_dataset.append(item)
                count += 1

    except FileNotFoundError:
        samples_list = None
        with open(eval_dataset_path, "r") as file:
            samples_list = json.load(file)

        for sample in samples_list:
            for citation_token in sample["bib_info"]:

                split_list = sample["paper"].split(citation_token)
                for i in range(len(split_list)-1):

                    context = split_list[i]

                    for token in sample["bib_info"]:
                        context = context.replace(token, "<|mask|>")

                    for possible_citation in sample["bib_info"][citation_token]:

                        citation_corpus_id = possible_citation["citation_corpus_id"]
                        if citation_corpus_id in sc_corpus_id_map:

                            sys.stdout.write("\033[F")
                            print(f"processing entry {count}")


                            rec = {
                                "context": context,
                                "citation_corpus_id": possible_citation["citation_corpus_id"]
                            }
                            eval_dataset.append(rec)
                            with open(out_file, "a") as outfile:
                                json.dump(rec, outfile)
                                outfile.write("\n")
                            count += 1

    print(f"eval dataset loaded")

    eval_indices = np.arange(len(eval_dataset))

    RECALL_K = 5

    if config["sc_retriever_topk"] < RECALL_K:
        raise ValueError(f"ScholarCopilot top-k ({config["sc_retriever_topk"]}) cannot be smaller than recall k ({RECALL_K})")

    shuffled_samples = True

    if shuffled_samples:
        random.shuffle(eval_indices)
    
    sc_rankings = [] # len_dataset x recall_k
    sc_pr_rankings = [] # len_dataset x recall_k
    gold = [] # len_dataset x 1
    max_samples = 100
    samples = 0
    num_fails = 0
    print()
    for i in eval_indices:
        sys.stdout.write("\033[F")
        print(f"processing entry {samples}")

        item = eval_dataset[i]

        context = item["context"]

        gold.append(item["citation_corpus_id"])

        sc_rankings.append(rank_with_scholarcopilot(
            context, index, lookup_indices, model, tokenizer, config
        ))

        ranking, fail = rank_with_scholarcopilot_with_passageretrieval(
            context, retrieval_dataset, sc_corpus_id_map, index, lookup_indices, model, tokenizer, passage_retrieval_models, config
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
