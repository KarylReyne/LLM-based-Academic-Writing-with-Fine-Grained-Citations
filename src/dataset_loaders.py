import sys
import ijson
import json
import re
import random
import numpy as np

from passage_retrieval_interface import apply_retrieval_context_window


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


def load_retrieval_dataset_from_sc_eval(retrieval_dataset_path, complete_dataset_path, sc_corpus_id_map, config):
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
        dict_list = None
        with open(complete_dataset_path, "r") as file:
            dict_list = json.load(file)

        for item in dict_list:
            sys.stdout.write("\033[F")
            print(f"processing entry {counter}")

            arxiv_id = item["arxiv_id"].split("-")[0]
            corpus_id = sc_corpus_id_map[arxiv_id]

            if complete_dataset_path == "data_train/scholar_copilot_train_data_500k.json":
                pattern = r'<\|cite_start\|>(.*?)<\|cite_end\|>'
                sentences = re.sub(pattern, config["citation_mask_token"], item["paper"])
                title = ""
                abstract = ""
                sections = [{
                    "title": "",
                    "sentences": [sentences]
                }]
            elif complete_dataset_path == "data_train/scholar_copilot_eval_data_1k.json":
                title = item["title"]
                abstract = item["abstract"]
                sections = [{
                    "title": "",
                    "sentences": [item["paper"]]
                }]
            else:
                raise ValueError(f"The dataset {complete_dataset_path} is not supported!")
            
            rec = {
                "corpus_id": corpus_id,
                "arxiv_id": arxiv_id,
                "title": title, 
                "abstract": abstract, 
                "sections": sections
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


def scholarcopilot_arxiv_to_corpus_id(path, sc_corpus_path):
    print("loading scholarcopilot_arxiv_to_corpus_id map...")
    id_map = {}
    try:
        with open(path, "r") as mapfile:
            id_map = json.load(mapfile)
    except FileNotFoundError:
        counter = 0
        print()
        with open(sc_corpus_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")
                id_map[item["paper_id"]] = item["corpus_id"]
                counter += 1
        with open(path, "w") as file:
            json.dump(id_map, file, ensure_ascii=False, indent=4)
    print("scholarcopilot_arxiv_to_corpus_id map loaded.")
    return id_map


def load_eval_dataset(eval_dataset_path, arxiv_to_corpus_id_map, tokenizer, config, shuffle=True, max_samples=None):
    print("loading evaluation dataset...")
    eval_dataset = []
    try:
        counter = 0
        print()
        with open(eval_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")
                eval_dataset.append(item)
                counter += 1
                if max_samples != None:
                    if counter >= max_samples:
                        break
    except FileNotFoundError:
        LAST_SENTENCE_ONLY = eval_dataset_path.replace("last_sentence", "") != eval_dataset_path

        input_file = "data/documents_3.0_with_ids.jsonl"
        if LAST_SENTENCE_ONLY:
            output_file = f"data/context_citation_pairs_last_sentence_documents_3.0.jsonl"
        else:
            output_file = f"data/context_citation_pairs_{config["query_context"]}_documents_3.0.jsonl"

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
                            # mask all citations in the context
                            for _entry in bib:
                                lefthand_context = lefthand_context.replace(f"#ref{_entry}#", config["citation_mask_token"])
                            pair = {
                                "context": lefthand_context,
                                "source_corpus_id": arxiv_to_corpus_id_map[source_arxiv_id],
                                "target_corpus_id": arxiv_to_corpus_id_map[target_arxiv_id]
                            }

                            if max_samples != None:
                                if num_pairs < max_samples:
                                    eval_dataset.append(pair)

                            with open(output_file, "a") as outfile:
                                json.dump(pair, outfile)
                                outfile.write("\n")
                            num_pairs += 1
                            
                    except KeyError:
                        pass # ignores targets that don't have an arxiv id

                counter += 1

    eval_indices = np.arange(len(eval_dataset))
    if shuffle:
        random.shuffle(eval_indices)
    print(f"evaluation dataset loaded.")
    return eval_dataset, eval_indices
