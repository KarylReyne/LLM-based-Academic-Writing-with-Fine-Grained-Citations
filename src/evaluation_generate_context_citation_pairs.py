import sys
import json
import os
import ijson

from passage_retrieval_interface import get_config, AutoTokenizer, apply_retrieval_context_window


if __name__ == "__main__":
    LAST_SENTENCE_ONLY = True

    config = get_config()
    tokenizer = AutoTokenizer.from_pretrained(config["retriever"])

    input_file = "data/documents_3.0_with_ids.jsonl"
    if LAST_SENTENCE_ONLY:
        output_file = f"data/context_citation_pairs_last_sentence_documents_3.0.jsonl"
    else:
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
            if not LAST_SENTENCE_ONLY:
                full_context = []
                for s in item["sections"]:
                    full_context.append(s["title"])
                    [full_context.append(sent) for sent in s["sentences"]]
                full_context = " ".join(full_context)

            bib = item["bibliography"]

            for entry in bib:
                try:
                    target_arxiv_id = bib[entry]["arxiv_id"] # this raises the exception
                    _ = arxiv_to_corpus_id_map[target_arxiv_id] # check which ids are in the dataset

                    source_arxiv_id = item["arxiv_id"]
                    entry_token = f"#ref{entry}#"

                    lefthand_context = None
                    if LAST_SENTENCE_ONLY:
                        for s in item["sections"]:
                            found = False
                            for sent in s["sentences"]:
                                split = sent.split(entry_token)
                                if len(split) >= 2:
                                    lefthand_context = split[0]
                                    found = True
                                    break # sent loop
                            if found:
                                break # s loop
                    else:
                        lefthand_context = full_context.split(entry_token)[0]
                        lefthand_context = apply_retrieval_context_window(lefthand_context, tokenizer, config)

                    if lefthand_context != None:
                        pair = {
                            "context": lefthand_context,
                            "source_corpus_id": arxiv_to_corpus_id_map[source_arxiv_id],
                            "target_corpus_id": arxiv_to_corpus_id_map[target_arxiv_id]
                        }
                        with open(output_file, "a") as outfile:
                            json.dump(pair, outfile)
                            outfile.write("\n")
                        num_pairs += 1
                        
                except KeyError:
                    pass # ignores targets that don't have an arxiv id

            counter += 1
