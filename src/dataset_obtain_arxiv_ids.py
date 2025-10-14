import json
import requests
import urllib, urllib.request
import sys
import ijson
import time


def get_arxiv_id_from_citation(citation, title_to_arxiv_id):
    arxiv_id = None
    id_type = None

    if len(citation.split(" abs/")) == 2: # direct extraction
        arxiv_id = citation.split(" abs/")[1].split(",")[0].split(" ")[0].rstrip(".")
        id_type = "extracted"

    elif len(citation.split("arXiv preprint arXiv:")) == 2: # direct extraction
        arxiv_id = citation.split("arXiv preprint arXiv:")[1].split(",")[0].split(" ")[0].rstrip(".")
        id_type = "extracted"

    else: # intra-dataset search
        title = ""
        for substring in citation.split(".")[1:]:
            if substring.startswith("  "): # titles seem to always be preceded by "  "
                title = substring.lstrip("  ")
                break
        title = title.lower().replace(" ", "")
        try:
            arxiv_id = title_to_arxiv_id[title]
            id_type = "from_dataset"
        except KeyError:
            pass

    if arxiv_id == None:
        raise ArxivSearchFailedError
        
    assert arxiv_id != None, citation
    arxiv_id = arxiv_id.split("v")[0] # remove version
    return arxiv_id, id_type


class ArxivSearchFailedError(Exception):
    """The arxiv title search did not yield the correct result"""
    pass

class NoResultMatchedError(Exception):
    """The arxiv title search did not yield the correct result"""
    pass


if __name__ == "__main__":
    dataset_path = "data/documents_3.0.json"
    new_dataset_path = f"{dataset_path.rstrip('.json')}_with_ids.jsonl"
    map_path = "data/title_to_arxiv_id_documents_3.0.json"
    FORCE_MAP_GENERATION = False

    title_to_arxiv_id = {}
    dataset = {}
    dataset_size = None
    try:
        if FORCE_MAP_GENERATION:
            raise FileNotFoundError
        with open(map_path, "r") as mapfile:
            title_to_arxiv_id = json.load(mapfile)
            dataset_size = len(title_to_arxiv_id.keys())
    except FileNotFoundError:
        with open(dataset_path, "r") as file:
            dataset_size = 0
            for line in file:
                rec = json.loads(line)
                dataset[rec["arxiv_id"]] = rec
                title = rec["title"].lower().replace(" ", "")
                title_to_arxiv_id[title] = rec["arxiv_id"]
                dataset_size += 1
        with open(map_path, "w") as file:
            json.dump(title_to_arxiv_id, file, ensure_ascii=False, indent=4)
    arxiv_id_dict = {v: k for k, v in title_to_arxiv_id.items()}
    print(f"title_to_arxiv_id loaded (unique titles: {dataset_size}/unique ids: {len(arxiv_id_dict.keys())}).")

    prev_processed_dataset = {}
    prev_processed_dataset_size = 0
    try:
        with open(new_dataset_path, "r") as file:
            for line in file:
                rec = json.loads(line)
                prev_processed_dataset[rec["arxiv_id"]] = rec
                prev_processed_dataset_size += 1
        print(f"prev_processed_dataset loaded ({prev_processed_dataset_size}).")
    except FileNotFoundError:
        pass

    ids = 0
    ids_in_dataset = 0
    num_extracted_ids = 0
    num_ids_from_dataset = 0
    num_searched_ids = 0
    search_fails = 0
    match_fails = 0

    for _ in range(8):
        print()

    counter = 1
    with open(dataset_path, "rb") as infile:
        for item in ijson.items(infile, "", multiple_values=True):

            arxiv_id = item["arxiv_id"]
            bib = item["bibliography"]

            for _ in range(8):
                sys.stdout.write("\033[F")
            print(f"processing sample {counter}/{dataset_size-prev_processed_dataset_size}")
            print(f"ids: {ids}")
            print(f"ids_in_dataset: {ids_in_dataset}")
            print(f"num_extracted_ids: {num_extracted_ids}")
            print(f"num_ids_from_dataset: {num_ids_from_dataset}")
            print(f"num_searched_ids: {num_searched_ids}")
            print(f"search_fails: {search_fails}")
            print(f"match_fails: {match_fails}")

            if arxiv_id in prev_processed_dataset:
                continue # skip already processed samples
            # if counter >= dataset_size:
            #     break

            for k in bib:
                citation = bib[k]
                bib[k] = {
                    "citation": citation
                }
                try:
                    retrieved_id, id_type = get_arxiv_id_from_citation(citation, title_to_arxiv_id)
                    bib[k] = {
                        "citation": citation,
                        "arxiv_id": retrieved_id
                    }
                    ids += 1
                    try:
                        tmp = arxiv_id_dict[retrieved_id]
                        ids_in_dataset += 1
                    except KeyError:
                        pass

                    if id_type == "extracted":
                        num_extracted_ids += 1
                    elif id_type == "from_dataset":
                        num_ids_from_dataset += 1
                    elif id_type == "searched":
                        num_searched_ids += 1

                except ArxivSearchFailedError:
                    search_fails += 1
                except NoResultMatchedError:
                    match_fails += 1

            with open(new_dataset_path, "a") as outfile:
                json.dump(item, outfile)
                outfile.write('\n')
            counter += 1
