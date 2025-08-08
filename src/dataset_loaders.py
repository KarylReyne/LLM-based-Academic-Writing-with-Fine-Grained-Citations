import sys
import ijson
import json
import re
import random
import numpy as np
from tqdm import tqdm

from passage_retrieval_interface import apply_retrieval_context_window
from sigterm_catcher import SIGTERMCatcher


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

            if complete_dataset_path == "scholarcopilot_data/scholar_copilot_train_data_500k.json":
                pattern = r'<\|cite_start\|>(.*?)<\|cite_end\|>'
                sentences = re.sub(pattern, config["citation_mask_token"], item["paper"])
                title = ""
                abstract = ""
                sections = [{
                    "title": "",
                    "sentences": [sentences]
                }]
            elif complete_dataset_path == "scholarcopilot_data/scholar_copilot_eval_data_1k.json":
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


def load_scholarcopilot_eval_dataset(eval_dataset_path, sc_eval_dataset_path, sc_arxiv_id_map, config, shuffle=True):
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
                        if citation_corpus_id in sc_arxiv_id_map:

                            sys.stdout.write("\033[F")
                            print(f"processing entry {count}")

                            rec = {
                                "context": context,
                                "source_arxiv_id": sample["paper_id"],
                                "target_corpus_id": citation_corpus_id
                            }
                            eval_dataset.append(rec)
                            with open(eval_dataset_path, "a") as outfile:
                                json.dump(rec, outfile)
                                outfile.write("\n")
                            count += 1

    eval_indices = np.arange(len(eval_dataset))
    if shuffle:
        random.shuffle(eval_indices)
    print(f"evaluation dataset loaded.")
    return eval_dataset, eval_indices


def load_pr_train_set_for_scholarcopilot(pr_train_dataset_path, docs_fulltext_dataset_path, docs_dataset_path, get_passage_from_context, tokenizer, config):

    def _get_passages_from_fulltext(fulltext, target_id, tokenizer, config):
        tokens = tokenizer(fulltext).to(config["scholarcopilot_device"])
        tokens = tokens["input_ids"] # get only the encoded tokens
        passages = []
        passage_idx = 0
        for i in range(0, len(tokens), config["passage_length"]):
            passage = tokenizer.decode(tokens[i:i+config["passage_length"]]).replace(config["tokenizer_begin_token"], "")
            psg_label = f"{target_id}_{passage_idx}" # without the section title bc it is not available !!!
            passages.append(psg_label+config["label_sep_token"]+passage)
            passage_idx += 1
        return passages

    print("loading training dataset...")
    train_dataset = []
    count = 0
    print()

    try:
        with open(pr_train_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {count}")
                train_dataset.append(item)
                count += 1
    
    except FileNotFoundError:
        corpus = {}
        try:
            with open(docs_fulltext_dataset_path, "rb") as file:
                for item in ijson.items(file, "", multiple_values=True):
                    sys.stdout.write("\033[F")
                    print(f"processing corpus entry {count}")

                    corpus[item["arxiv_id"]] = item
                    count += 1


        except FileNotFoundError: # create a smaller, more manageable dataset
            with open(docs_dataset_path, "rb") as file:
                for item in ijson.items(file, "", multiple_values=True):
                    sys.stdout.write("\033[F")
                    print(f"processing corpus entry {count}")
                    
                    fulltext = item["title"]
                    fulltext += " ".join(item["abstract"])
                    for section in item["sections"]:
                        fulltext += section["title"]
                        fulltext += " ".join(section["sentences"])
                    
                    new_item = {
                        "arxiv_id": item["arxiv_id"],
                        "fulltext": fulltext,
                        "bibliography": item["bibliography"]
                    }

                    corpus[item["arxiv_id"]] = new_item

                    with open(docs_fulltext_dataset_path, "a") as outfile:
                        json.dump(new_item, outfile)
                        outfile.write("\n")
                    count += 1

        print("building train dataset...\n")        
        catcher = SIGTERMCatcher()
        while not catcher.was_killed:

            count = 0
            skipped = 0
            for arxiv_id in tqdm(corpus):
                try:
                    item = corpus[arxiv_id]

                    sys.stdout.write("\033[F")
                    print(f"processing entry {count} ({skipped} skipped)")
                    
                    # num citations for training the retriever per paper
                    # same as SC
                    num_targets = 4 
                    # how many citations are preserved in the fulltext paper
                    # affects generator training
                    # cuts down processing time per paper considerably
                    max_paper_refs = 10*num_targets
                    assert max_paper_refs >= num_targets

                    # picks citations to use for training randomly
                    bib = item["bibliography"]
                    bib_items = list(bib.items())
                    bib_items_indices = np.arange(len(bib_items))
                    random.shuffle(bib_items_indices)
                    
                    replaceable_citations = []
                    for i in bib_items_indices:
                        k, v = bib_items[i]
                        if "arxiv_id" in v:
                            if v["arxiv_id"] in corpus: # guarantees that the paper is retrievable
                                replaceable_citations.append(k)
                                if len(replaceable_citations) >= max_paper_refs:
                                    break

                    all_targets = []

                    paper = item["fulltext"]
                    for key in replaceable_citations:
                        target_id = bib[key]["arxiv_id"]
                        # SC
                        # target = corpus[target_id]["title"]+":"
                        # target += " ".join(corpus[target_id]["abstract"]).lstrip(" Abstract")
                        # PR
                        target = get_passage_from_context(
                            paper.split(f"#ref{key}#")[0], 
                            [{
                                "arxiv_id": "",
                                "passages": _get_passages_from_fulltext(
                                corpus[target_id]["fulltext"], target_id, tokenizer, config
                            )}]
                        )
                        all_targets.append(target)
                        target = f"<|cite_start|> (Reference: {target}) <|cite_end|>"
                        paper = paper.replace(f"#ref{key}#", target)

                    random_indices = np.arange(len(replaceable_citations))
                    random.shuffle(random_indices)
                    if len(random_indices) < num_targets:
                        raise ValueError(f"not enough replaceable_citations ({len(replaceable_citations)}) for the given num_targets ({num_targets})")

                    targets = []
                    targets_idx = []
                    for i in random_indices:
                        target = all_targets[i]
                        target = f"<|reference_start|> {target} <|reference_end|>"
                        targets.append(target)
                        targets_idx.append(int(i))
                        if len(targets) >= num_targets:
                            break

                    rec = {
                        "paper": paper,
                        "targets": targets,
                        "targets_idx": targets_idx
                    }
                    with open(pr_train_dataset_path, "a") as outfile:
                        json.dump(rec, outfile)
                        outfile.write("\n")
                    count += 1

                except Exception as e:
                    print(f"Failure due to {type(e)}: {e.args}\n")
                    skipped += 1
                    continue

        # caught SIGKILL
        print("caught SIGKILL")
        exit()
            

def load_sections_eval_dataset(target_sections, sections_eval_dataset_path, docs_dataset_path, docs_id_map, sc_id_map, max_samples=1000):
    print("loading sections eval dataset...")
    contains_substring = lambda s, sub: s.lower() != s.lower().replace(sub.lower(), "")
    eval_dataset = []
    count = 0
    print()

    try:
        with open(sections_eval_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {count}")
                eval_dataset.append(item)
                count += 1

    except FileNotFoundError:
        num_samples = 0
        skipped = 0
        with open(docs_dataset_path, "rb") as file:
            for item in ijson.items(file, "", multiple)
                sys.stdout.write("\033[F")
                print(f"processing entry {count} - found {num_samples}/{max_samples} ({skipped} skipped)")

                # identify relevant sections
                sections_fulltext = ""
                for section in item["sections"]:
                    for target_section in target_sections:
                        if contains_substring(section["title"], target_section):
                            sections_fulltext += section["title"]
                            sections_fulltext += " ".join(section["sentences"])
                
                if sections_fulltext == "":
                    skipped += 1
                    continue

                # identify retrievable citations
                # create eval sample from each citation