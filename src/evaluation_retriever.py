import json
import sys
import itertools

from passage_retrieval_interface import get_config, get_passage_retrieval_models, retrieve_relevant_passages
from util import recall_at_k
from latex_parsing import identify_citations_in_source_doc, search_arxiv_for_citations_data


def get_query_context(source_paper, passage_retrieval_models, config):
    generated_contexts = []
    # TODO
    # tokenize
    # identify citation indices
    # get ctx for each cit idx
    return generated_contexts


def load_dataset(dataset_path):
    dataset = {}
    bib_to_paper_id = {}
    meta_data = None
    with open(dataset_path, "r") as file:
        meta_data = json.load(file)
    for corpus_id in meta_data:
        curr = meta_data[corpus_id]
        fulltext = curr["fulltext"]
        if fulltext != "<|tex_parsing_error|>":
            bib_id = curr["bibtex"].split(",")[0].lstrip("@article{")
            _, citations_data_ids = search_arxiv_for_citations_data(curr["paper_id"])
            dataset[curr["paper_id"]] = {
                "fulltext" : fulltext,
                "bib_id": bib_id,
                "citations_data_ids": citations_data_ids
            }
            bib_to_paper_id[bib_id] = curr["paper_id"] # reverse of dataset[paper_id]["bib_id"]
    return dataset, bib_to_paper_id


if __name__ == "__main__":
    RECALL_K = 1

    config = get_config()
    passage_retrieval_models = get_passage_retrieval_models(config)

    source_paper_ids = []
    # dataset[paper_id] = {fulltext,bib_id}
    dataset, bib_to_paper_id = load_dataset("scholarcopilot_data/corpus_data_arxiv_1215_fulltext.jsonl")
    print(dataset[:5])
    reference_ids = [id for id in dataset]
    print(f"loaded dataset with {len(dataset)} documents")

    # TODO - WIP
    source_paper_ids.append(reference_ids[0])
    target_bib_ids = [id for id in bib_to_paper_id]

    # zip
    source_paper_ids = list(itertools.chain.from_iterable([
        list(itertools.repeat(id, len(target_bib_ids))) for id in source_paper_ids
    ]))
    target_bib_ids = list(itertools.chain.from_iterable(itertools.repeat(target_bib_ids, len(source_paper_ids))))
    assert len(source_paper_ids) == len(target_bib_ids), f"{len(source_paper_ids)} != {len(target_bib_ids)}"
    zip_ids = zip(source_paper_ids, target_bib_ids)
    print(f"generated {len(zip_ids)} eval samples")

    # TODO: this yields 2236154944 samples!!! - only use bib ids that appear in a given src doc !

    # rankings = [[high,...,low],...] for every src_id/tgt_id/ctx combination
    rankings = []
    gold_ids = []

    print()
    for src_id, tgt_id in zip_ids:
        source_paper = dataset[src_id]["fulltext"]

        # generated_contexts = get_query_context(source_paper, passage_retrieval_models, config)
        _, generated_contexts = identify_citations_in_source_doc(
            source_paper, tgt_id, passage_retrieval_models["retr_tokenizer"], config
        )

        for generated_context in generated_contexts:
            print(generated_context)
            _, _, _, final_scores = retrieve_relevant_passages(
                generated_context, reference_ids, passage_retrieval_models, config
            )

            rankings.append([l.split("_")[0] for l in final_scores]) # paper_ids
            gold_ids.append(bib_to_paper_id[tgt_id])

            sys.stdout.write("\033[F")
            print(f"{len(rankings)} so far")


    print(f"matched {len(rankings)} citations in the dataset")
    
    eval_score = recall_at_k(rankings, gold_ids, k=RECALL_K)
    print(f"recall@{RECALL_K}: {eval_score}")

        