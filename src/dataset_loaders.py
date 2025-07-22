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


def load_retrieval_dataset_for_sc_corpus_with_fulltext(
    retrieval_dataset_path, # name of the new dataset
    documents_retrieval_dataset, # documents 3.0 retrieval dataset
    sc_retrieval_corpus_path, # scholarcopilot retrieval corpus path
    documents_corpus_id_map, # documents 3.0 id map
    config
):
    print("loading retrieval dataset...")
    retrieval_dataset = {}
    print()
    counter = 0
    match_counter = 0
    try:
        with open(retrieval_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {counter}")

                retrieval_dataset[item["sc_corpus_id"]] = item
                counter += 1

    except FileNotFoundError:
        print()
        counter = 0
        match_counter = 0
        with open(sc_retrieval_corpus_path, "rb") as file: # scholarcopilot retrieval corpus
            for item in ijson.items(file, "", multiple_values=True):
                
                sys.stdout.write("\033[F")
                print(f"processing entry {match_counter}/{counter} (matched/total)")

                sc_corpus_id = item["corpus_id"]
                target_arxiv_id = item["paper_id"]

                try: 
                    docs_corpus_id = documents_corpus_id_map[target_arxiv_id] # arxiv_id in documents 3.0?
                    docs_item = documents_retrieval_dataset[docs_corpus_id]
                
                    rec = {
                        "sc_corpus_id": sc_corpus_id,
                        "arxiv_id": target_arxiv_id,
                        "title": docs_item["title"], 
                        "abstract": docs_item["abstract"], 
                        "sections": docs_item["sections"]
                    }
                    retrieval_dataset[sc_corpus_id] = rec

                    match_counter += 1
                    counter += 1

                    with open(retrieval_dataset_path, "a") as outfile:
                        json.dump(rec, outfile)
                        outfile.write("\n")

                except KeyError:
                    counter += 1

    print("retrieval dataset loaded.")
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

    if not eval_dataset_path.endswith("documents_3.0.jsonl"):
        raise ValueError("load_eval_dataset only works for the documents_3.0 dataset!")

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
        INTRO_RELWORK_ONLY = eval_dataset_path.replace("intro+relwork", "") != eval_dataset_path

        input_file = "data/documents_3.0_with_ids.jsonl"
        output_file = eval_dataset_path

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
                if LAST_SENTENCE_ONLY:
                    pass                    
                elif INTRO_RELWORK_ONLY:
                    for s in item["sections"]:
                        low_title = s["title"].lower()
                        if low_title != low_title.replace("introduction", "") or low_title != low_title.replace("related work", ""):
                            full_context.append(s["title"])
                            [full_context.append(sent) for sent in s["sentences"]]
                    full_context = " ".join(full_context) 
                else:
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

                        lefthand_context = []
                        if LAST_SENTENCE_ONLY:
                            for s in item["sections"]:
                                for sent in s["sentences"]:
                                    split = sent.split(entry_token)
                                    if len(split) >= 2:
                                        for ctx in split[:len(split)-1]:
                                            lefthand_context.append(apply_retrieval_context_window(ctx, tokenizer, config))
                        elif INTRO_RELWORK_ONLY:
                            if len(full_context.split(entry_token)) >= 2: # bc not all bib entries are cited in intro+relwork
                                split = full_context.split(entry_token)
                                for ctx in split[:len(split)-1]:
                                    lefthand_context.append(apply_retrieval_context_window(ctx, tokenizer, config))
                        else:
                            split = full_context.split(entry_token)
                            for ctx in split[:len(split)-1]:
                                lefthand_context.append(apply_retrieval_context_window(ctx, tokenizer, config))

                        for ctx in lefthand_context:
                            # mask all citations in the context
                            for _entry in bib:
                                ctx = ctx.replace(f"#ref{_entry}#", config["citation_mask_token"])

                            pair = {
                                "context": ctx,
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


def load_scholarcopilot_eval_dataset(eval_dataset_path, sc_eval_dataset_path, config, shuffle=True):
    eval_dataset = []
    count = 0
    print()
    try:
        with open(eval_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {count}")
                eval_dataset.append(item)
                count += 1

    except FileNotFoundError:
        samples_list = None
        with open(sc_eval_dataset_path, "r") as file:
            samples_list = json.load(file)

        for sample in samples_list:
            for citation_token in sample["bib_info"]:

                split_list = sample["paper"].split(citation_token)
                for i in range(len(split_list)-1):

                    context = split_list[i]

                    for token in sample["bib_info"]:
                        context = context.replace(token, config["citation_mask_token"])

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

    eval_indices = np.arange(len(eval_dataset))
    if shuffle:
        random.shuffle(eval_indices)
    print(f"evaluation dataset loaded.")
    return eval_dataset, eval_indices
