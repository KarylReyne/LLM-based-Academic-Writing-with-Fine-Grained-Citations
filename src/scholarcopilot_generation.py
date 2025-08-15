from datetime import datetime
import tempfile
from scholarcopilot_model import *
import torch
import faiss
import time
import tarfile
import passage_reranking
from passage_retrieval_interface import *


def split_yield_list(input_text, prefix_length):
    prefix_text = input_text[:prefix_length]
    text = input_text[prefix_length:]
    text_list = text.split(" ")
    return prefix_text, text_list


def stream_generate(text, citations_data, index, lookup_indices, model, tokenizer, retrieval_dataset, arxiv_to_corpus_id_map, passage_retrieval_models, config, do_passage_retrieval=True, silent=False):
    sentence_num = 0
    enough = False
    current_text = text
    current_text = preprocess_input_text(current_text)
    display_text = current_text.replace("<|paper_start|> ", "")
    curr_prefix_length = len(display_text)
    current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, config["scholarcopilot_device"], current_text)
    unique_reference_id_list = [] # (arxiv_id, suffix)
    display_text, new_citation_data = replace_citations(
        current_text, unique_reference_id_list, retrieval_dataset, arxiv_to_corpus_id_map
    )
    citations_data += new_citation_data
    curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
    # print("curr_yield_text, yield_list", curr_yield_text, yield_list)
    for each in yield_list:
        if "." in each and (each.endswith(".") or ".\n" in each):
            sentence_num += 1
            if not silent:
                print("sentence_num: ", sentence_num, "each", each)
        curr_yield_text += " " + each
        yield curr_yield_text, citations_data
        time.sleep(0.1)
    curr_prefix_length = len(curr_yield_text)

    while cite_start_hidden_state is not None and not enough:
        retrieved_k_results = retrieve_reference(
            index, lookup_indices, cite_start_hidden_state, config, top_k=config["sc_retriever_topk"]
        )
        references, distances = collect_retrieval_results(retrieved_k_results, retrieval_dataset)

        # --- BEGIN passage retrieval ---
        start = time.time()
        if do_passage_retrieval:
            try:
                reranking_results = retrieve_relevant_passages(
                    current_text, references, passage_retrieval_models, config
                )
                best_matching_passage = reranking_results["ranked_passages"][0]
                best_passage_label = reranking_results["ranked_passage_labels"][0]
                best_passage_score = reranking_results["ranked_passage_scores"][0]
                best_reference_arxiv_id = best_passage_label.split("_")[0]
                best_matching_passage = best_matching_passage+"<|cite_end|>"
                if not silent:
                    print("best matching passage: ", best_matching_passage)
            except passage_reranking.InvalidLLMResponseError:
                best_matching_passage = references[0]["abstract"]+"<|cite_end|>" # default to abstract if llm response parsing failed
                best_reference_arxiv_id = references[0]["arxiv_id"]
                if not silent:
                    print("tex or llm response parsing failed, using abstract as reference: ", best_matching_passage)
        else:
            best_matching_passage = references[0]["abstract"]+"<|cite_end|>"
            best_reference_arxiv_id = references[0]["arxiv_id"]
            if not silent:
                print("best abstract as reference: ", best_matching_passage)
        if not silent:
            print(f"best reference after passage retrieval: {best_reference_arxiv_id}")
            print("***************Retrieval cost (time): ", time.time() - start)
        # --- END passage retrieval ---

        unique_id_suffix = len(unique_reference_id_list) # this resolves duplicate arxiv_ids in the list of references
        unique_reference_id_list.append((best_reference_arxiv_id, unique_id_suffix)) # (arxiv_id, suffix)

        current_text = current_text + best_matching_passage

        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, config["scholarcopilot_device"], current_text)
        display_text, new_citation_data = replace_citations(
            current_text, unique_reference_id_list, retrieval_dataset, arxiv_to_corpus_id_map
        )

        citations_data += new_citation_data

        # add passage retrieval-specific entries to citations data
        if do_passage_retrieval:
            # get the data entry of the newly added citation
            citation_dict = citations_data[-1]
            # check that its the correct entry
            assert citation_dict["citation_key"] == f"arxivID-{best_reference_arxiv_id}-{unique_id_suffix}"
            # add passage retrieval result
            citation_dict["matched_passage"] = best_matching_passage.rstrip("<|cite_end|>")
            citation_dict["passage_label"] = best_passage_label
            citation_dict["passage_score"] = best_passage_score
            # save modified data entry 
            citations_data[-1] = citation_dict

        curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
        # print("curr_yield_text, yield_list", curr_yield_text, yield_list)
        for each in yield_list:
            if "." in each and (each.endswith(".") or ".\n" in each):
                sentence_num += 1
                print("sentence_num: ", sentence_num, "each", each)
            curr_yield_text += " " + each
            yield curr_yield_text, citations_data
            time.sleep(0.1)
        curr_prefix_length = len(curr_yield_text)

    display_text, new_citation_data = post_process_output_text(
        display_text, unique_reference_id_list, retrieval_dataset, arxiv_to_corpus_id_map
    )
    citations_data += new_citation_data
    yield display_text, citations_data
    time.sleep(0.1)


def load_example(file_path=""):
    example_text = ""
    with open(file_path, "r") as fi:
        for line in fi.readlines():
            example_text += line
    return example_text


if __name__ == "__main__":
    config = get_config()

    model_path = "scholarcopilot_model_v1208/"
    model, tokenizer = load_model(model_path, config)

    passage_retrieval_models = get_passage_retrieval_models(config)
    print("models loaded")

    id_map_path = "data/arxiv_to_corpus_id_documents_3.0.json"
    processed_corpus_path = "data/documents_3.0_processed_corpus.jsonl"
    arxiv_to_corpus_id_map = arxiv_to_corpus_id(id_map_path, processed_corpus_path)
    
    retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    complete_dataset_path = "data/documents_3.0_with_ids.jsonl"
    retrieval_dataset = load_retrieval_dataset(retrieval_dataset_path, complete_dataset_path, arxiv_to_corpus_id_map)

    index_dir = "data/index"
    lookup_indices_dir = "data/lookup_indices.npy"
    index, lookup_indices = load_faiss_index(index_dir, lookup_indices_dir)
    print("index building finished")

    citations_data = []

    # starting left-side context for the generation model
    example_path = "scholarcopilot_examples/vlm2vec-example.txt"
    text_input = load_example(example_path)

    print("pre-generation text_input:", text_input)

    gen = stream_generate(
        text_input, citations_data, index, lookup_indices, model, tokenizer, # scholar copilot
        retrieval_dataset, arxiv_to_corpus_id_map, passage_retrieval_models, config # passage retrieval
    )
    for t in gen:
        text_input, citations_data = t
    print("text_input:", text_input)
    
    save_results({
        "given generation input": example_path,
        "generated paper": text_input,
        "citations_data": citations_data
    }, config, mode="generation")


