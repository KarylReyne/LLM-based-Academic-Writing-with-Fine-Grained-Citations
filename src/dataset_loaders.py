import sys
import ijson
import json



def load_retrieval_dataset(retrieval_dataset_path, complete_dataset_path, id_map):
    print("loading retrieval dataset...")
    retrieval_dataset = {}
    print()
    counter = 0
    try:
        with open(retrieval_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")
                retrieval_dataset[item["corpus_id"]] = item
                counter += 1
    except FileNotFoundError:
        with open(complete_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")
                arxiv_id = item["arxiv_id"]
                corpus_id = id_map[arxiv_id]
                rec = {
                    "corpus_id": corpus_id,
                    "arxiv_id": arxiv_id,
                    "title": item["title"], 
                    "abstract": " ".join(item["abstract"]), 
                    "sections": item["sections"]
                }
                retrieval_dataset[corpus_id] = rec
                counter += 1
                with open(retrieval_dataset_path, "a") as outfile:
                    json.dump(rec, outfile)
                    outfile.write("\n")
    print("retrieval dataset loaded.")
    return retrieval_dataset


def load_prebuilt_passages_dataset(retrieval_dataset_path, complete_dataset_path, id_map, tokenizer, config):
    print("loading prebuilt passages dataset...")
    retrieval_dataset = {}
    print()
    counter = 0
    try:
        with open(retrieval_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")
                retrieval_dataset[item["corpus_id"]] = item
                counter += 1
    except FileNotFoundError:
        with open(complete_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")
                arxiv_id = item["arxiv_id"]
                corpus_id = id_map[arxiv_id]
                title = item["title"]
                abstract = " ".join(item["abstract"])
                passages = []
                for section in item["sections"]:
                    section_label = f"{arxiv_id}_{title.lstrip(" ").replace(" ", "-")}"
                    text = " ".join(section["sentences"])
                    tokens = tokenizer(text).to(config["retriever_device"])
                    tokens = tokens["input_ids"] # get only the encoded tokens

                    passage_idx = 0
                    for i in range(0, len(tokens), config["passage_length"]):
                        passage_label = f"{section_label}-{passage_idx}"
                        passage = tokenizer.decode(tokens[i:i+config["passage_length"]]).replace(config["tokenizer_begin_token"], "")
                        passages.append(passage_label+config["label_sep_token"]+passage)
                        passage_idx += 1
                rec = {
                    "corpus_id": corpus_id,
                    "arxiv_id": arxiv_id,
                    "title": title, 
                    "abstract": abstract,
                    "passages": passages
                }
                retrieval_dataset[corpus_id] = rec
                counter += 1
                with open(retrieval_dataset_path, "a") as outfile:
                    json.dump(rec, outfile)
                    outfile.write("\n")
    print("prebuilt passages dataset loaded.")
    return retrieval_dataset


def arxiv_to_corpus_id(path, processed_corpus_path):
    print("loading arxiv_to_corpus_id map...")
    id_map = {}
    try:
        with open(path, "r") as mapfile:
            id_map = json.load(mapfile)
    except FileNotFoundError:
        counter = 0
        print()
        with open(processed_corpus_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")
                id_map[item["arxiv_id"]] = item["corpus_id"]
                counter += 1
        with open(path, "w") as file:
            json.dump(id_map, file, ensure_ascii=False, indent=4)
    print("arxiv_to_corpus_id map loaded.")
    return id_map