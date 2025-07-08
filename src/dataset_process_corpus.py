import sys
import json
import os
import ijson


def process_single_dataset_entry(item, corpus_id):

    result = {
        "corpus_id": corpus_id,
        "arxiv_id": item["arxiv_id"],
        "title": item["title"],
        "abstract": " ".join(item["abstract"])
    }

    # maybe add fulltext as well?

    return result


if __name__ == "__main__":
    input_file = "data/documents_3.0_with_ids.jsonl"
    output_file = "data/documents_3.0_processed_corpus.jsonl"

    print()
    counter = 0
    with open(input_file, "rb") as infile:
        for item in ijson.items(infile, "", multiple_values=True):

            sys.stdout.write("\033[F")
            print(f"processing paper-{counter}")

            corpus_id = f"paper-{counter}"
            curr_corpus_item = process_single_dataset_entry(item, corpus_id)
            
            with open(output_file, "a") as outfile:
                json.dump(curr_corpus_item, outfile)
                outfile.write('\n')
            counter += 1
