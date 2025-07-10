import sys
import json
import os
import ijson


if __name__ == "__main__":
    input_file = "data/documents_3.0_with_ids.jsonl"
    output_file = "data/arxiv_id_to_citation_string_documents_3.0.jsonl"
    arxiv_ids_map_file = "data/arxiv_to_corpus_id_documents_3.0.json"

    arxiv_ids_map = {}
    with open(arxiv_ids_map_file, "r") as file:
        arxiv_ids_map = json.load(file)

    print("\n")
    matched_papers = 0
    counter = 0
    citations_map = {}
    with open(input_file, "rb") as infile:
        for item in ijson.items(infile, "", multiple_values=True):
            
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            print(f"matched papers: {matched_papers}")
            print(f"processed dataset entries: {counter}")

            bib = item["bibliography"]

            for entry in bib:
                try:
                    arxiv_id = bib[entry]["arxiv_id"] # this raises the exception
                    if arxiv_id in arxiv_ids_map: # check which ids are in the dataset
                        if not arxiv_id in citations_map: # check if this id is new
                            matched_papers += 1
                            citations_map[arxiv_id] = bib[entry]["citation"]
                except KeyError:
                    pass
            counter += 1
            
    with open(output_file, "w") as outfile:
        json.dump(citations_map, outfile, ensure_ascii=False, indent=4)
