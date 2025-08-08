import tqdm
import ijson
import torch
import pickle
import h5py
import sys
import os
import numpy as np
import json
import tqdm

from passage_retrieval_interface import get_config, get_passage_retrieval_models, retrieve_relevant_passages, apply_retrieval_context_window, get_candidate_passages
from passage_retrieval_instructions import retrieval_instruction_document
from dataset_loaders import load_retrieval_dataset, arxiv_to_corpus_id


def _get_passages_data_items(retr_dataset_entry, tokenizer, config):
    candidate_passages = get_candidate_passages(
        [retr_dataset_entry], tokenizer, config
    )
    passage_ids = [] # arxiv.id_section-title-passageidx
    passages = []
    for p in candidate_passages:
        split = p.split(config["label_sep_token"])
        passage_ids.append(split[0])
        passages.append(split[1])
    return passage_ids, passages


def main():
    config = get_config()
    passage_retrieval_models = get_passage_retrieval_models(config)

    docs_id_map_path = "data/arxiv_to_corpus_id_documents_3.0.json"
    docs_id_map = arxiv_to_corpus_id(docs_id_map_path, "data/documents_3.0_processed_corpus.jsonl")

    retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    docs_dataset_path = "data/documents_3.0_with_ids_only.jsonl"
    retrieval_dataset = load_retrieval_dataset(retrieval_dataset_path, docs_dataset_path, docs_id_map)

    encodings_output_path = f"data/documents_3.0_{config["passage_length"]}-passages_{config["retriever"]}-encoded.pkl"
    passages_data_path = encodings_output_path.replace(f"{config["retriever"]}-encoded.pkl", "passages-data.jsonl")

    # load/create passages
    passages_data = []
    count = 0
    print()
    try:
        with open(passages_data_path, "rb") as file:
            for item in ijson.items(file, "", multiple_values=True):
                sys.stdout.write("\033[F")
                print(f"processing entry {count}")
                passages_data.append((item["passage_ids"], item["passages"]))
                count += 1

    except FileNotFoundError:
        for corpus_id in retrieval_dataset:
            sys.stdout.write("\033[F")
            print(f"processing entry {count}")
            passage_ids, passages = _get_passages_data_items(
                retrieval_dataset[corpus_id], passage_retrieval_models["retr_tokenizer"], config
            )
            passages_data.append((passage_ids, passages))

            with open(passages_data_path, "a") as outfile:
                json.dump({
                    "passage_ids": passage_ids,
                    "passages": passages
                }, outfile)
                outfile.write("\n")

            count += 1

    # encode with reasonir
    max_shards = int(np.ceil([len(passages_data)/samples_per_shard])[0])

    starting_passages_idx = 0
    for i in range(1, max_shards+1):
        this_shard_path = encodings_output_path.replace(".pkl", f"_shard-{i}-of-{max_shards}.pkl")
        if os.path.isfile(this_shard_path):
            starting_passages_idx += samples_per_shard
            print(f"found shard: {this_shard_path}")
        else:
            break

    print(f"starting from {starting_passages_idx}")
    encoded = []
    lookup_indices = []
    num_shards = 1
    samples_per_shard = 30000
    curr_num_samples = 0
    PSG_DATA_BATCH_SIZE = 128 # how many samples are passed to the encoder at once
    for i in tqdm.trange(starting_passages_idx, len(passages_data), PSG_DATA_BATCH_SIZE):
        passage_ids = []
        passages = []
        batch_data = passages_data[i:i+PSG_DATA_BATCH_SIZE]
        curr_num_samples += len(batch_data)
        for (ids, psg) in batch_data:
            passage_ids.extend(ids)
            passages.extend(psg)

        lookup_indices.extend(passage_ids)
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                reps = passage_retrieval_models["retriever"].encode(
                    passages,
                    instruction=retrieval_instruction_document,
                    batch_size=config["retr_batch_size"],
                    max_length=config["passage_length"]
                )
                encoded.extend(reps)

        print(f"num samples in this shard: {curr_num_samples}/{samples_per_shard}")
        if curr_num_samples >= samples_per_shard:
            this_shard_path = encodings_output_path.replace(".pkl", f"_shard-{num_shards}-of-{max_shards}.pkl")
            print(f"saving shard: {this_shard_path}")
            with open(this_shard_path, 'wb') as f:
                pickle.dump((encoded, lookup_indices), f)
            encoded = []
            lookup_indices = []
            curr_num_samples = 0
            num_shards += 1

    with open(encodings_output_path.replace(".pkl", f"_shard-{num_shards}-of-{max_shards}.pkl"), 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)

    # convert to h5py dataset
    encoded = []
    lookup_indices = []
    for shard_idx in tqdm.trange(1, max_shards+1):
        this_shard_path = encodings_output_path.replace(".pkl", f"_shard-{shard_idx}-of-{max_shards}.pkl")
        with open(this_shard_path, "rb") as infile:
            (e, li) = pickle.load(infile)
            encoded.extend(e)
            lookup_indices.extend(li)
    try:
        h5_path = this_shard_path.replace('.pkl', '.h5')
        save_prefix = encodings_output_path.lstrip("data/").replace(".pkl", "_")
        with h5py.File(h5_path, 'w') as outfile:
            outfile.create_dataset(save_prefix+'encoded', data=encoded)
            outfile.create_dataset(save_prefix+'lookup_indices', data=lookup_indices)
    except Exception as e:
        print(f"Failed to save encoded data to HDF5: {e}")


if __name__ == "__main__":
    main()

