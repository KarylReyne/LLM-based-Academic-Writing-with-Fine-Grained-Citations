import sys
import json
import os
import ijson

from passage_retrieval_interface import get_config, AutoTokenizer, apply_retrieval_context_window


if __name__ == "__main__":

    config = get_config()
    # tokenizer = AutoTokenizer.from_pretrained(config["retriever"])

    input_file = "data/documents_3.0_with_ids.jsonl"
    output_file = f"data/context_citation_pairs_{config["query_context"]}_documents_3.0.jsonl"
    arxiv_ids_map_file = "data/arxiv_to_corpus_id_documents_3.0.json"

    # arxiv id -> corpus id
    arxiv_to_corpus_id_map = {}
    with open(arxiv_ids_map_file, "r") as file:
        arxiv_to_corpus_id_map = json.load(file)

    print("\n")
    num_pairs = 0
    counter = 0
    with open(input_file, "rb") as infile:
        for item in ijson.items(infile, "", multiple_values=True):
            
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            print(f"generated pairs: {num_pairs}")
            print(f"processed dataset entries: {counter}")

            full_context = []
            for s in item["sections"]:
                full_context.append(s["title"])
                [full_context.append(sent) for sent in s["sentences"]]
            full_context = " ".join(full_context)

            bib = item["bibliography"]

            for entry in bib:
                try:
                    target_arxiv_id = bib[entry]["arxiv_id"] # this raises the exception
                    if target_arxiv_id in arxiv_to_corpus_id_map: # check which ids are in the dataset

                        source_arxiv_id = item["arxiv_id"]
                        entry_token = f"#ref{entry}#"
                        lefthand_context = full_context.split(entry_token)[0]
                        lefthand_context = apply_retrieval_context_window(lefthand_context, tokenizer, config)
                        pair = {
                            "context": lefthand_context,
                            "source_corpus_id": arxiv_to_corpus_id_map[source_arxiv_id],
                            "target_corpus_id": arxiv_to_corpus_id_map[target_arxiv_id]
                        }
                        num_pairs += 1
                except KeyError:
                    pass # ignores targets that don't have an arxiv id
                
                with open(output_file, "w") as outfile:
                    json.dump(pair, outfile, ensure_ascii=False, indent=4)

            counter += 1
