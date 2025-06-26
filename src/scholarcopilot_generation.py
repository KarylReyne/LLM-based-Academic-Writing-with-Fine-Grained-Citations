from datetime import datetime
import tempfile
from scholarcopilot_model import *
import torch
import faiss
import time
import tarfile
import passage_reranking

from passage_retrieval_interface import *
from latex_parsing import TexParsingError


def split_yield_list(input_text, prefix_length):
    prefix_text = input_text[:prefix_length]
    text = input_text[prefix_length:]
    text_list = text.split(" ")
    return prefix_text, text_list


def stream_generate(text, citations_data, passage_retrieval_models, config):
    sentence_num = 0
    enough = False
    current_text = text
    current_text = preprocess_input_text(current_text)
    display_text = current_text.replace("<|paper_start|> ", "")
    curr_prefix_length = len(display_text)
    current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
    reference_id_list = []
    display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
    citations_data += citation_data_list
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
    while cite_start_hidden_state is not None and not enough:
        retrieved_k_results = retrieve_reference(
            index, lookup_indices, cite_start_hidden_state, top_k=config["sc_retriever_topk"]
        )
        references, reference_ids = llm_rerank(retrieved_k_results, meta_data)

        # --- BEGIN passage retrieval ---
        generated_context = current_text
        if not config["retrieve_over_multiple_documents"]:
            references = [references[0]]
            reference_ids = [reference_ids[0]]
        tex_parsing_failed = False
        try:
            best_matching_passage, best_passage_label, best_passage_score = retrieve_relevant_passages(
                generated_context, reference_ids, passage_retrieval_models, config
            )
            best_reference_id = best_passage_label.split("_")[0]
            best_matching_passage = best_matching_passage+"<|cite_end|>"
            print("best matching passage: ", best_matching_passage)
        except TexParsingError or passage_reranking.InvalidLLMResponseError:
            tex_parsing_failed = True
            best_matching_passage = references[0] # default to standart ScholarCopilot if tex or llm response parsing failed
            best_reference_id = reference_ids[0]
            print("tex or llm response parsing failed, using abstract as reference: ", best_matching_passage)
        # --- END passage retrieval ---

        reference_id_list.append(best_reference_id)

        # current_text = current_text + reference
        current_text = current_text + best_matching_passage

        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
        display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)

        # citations_data += citation_data_list
        ids = [d["paper_id"] for d in citation_data_list]
        citation_index = ids.index(best_reference_id)
        citation_dict = citation_data_list[citation_index]
        if tex_parsing_failed:
            citation_dict["matched_passage"] = "<|tex_parsing_failed|>"
            citation_dict["passage_label"] = "<|tex_parsing_failed|>"
            citation_dict["passage_score"] = "<|tex_parsing_failed|>"
        else:
            citation_dict["matched_passage"] = best_matching_passage.rstrip("<|cite_end|>")
            citation_dict["passage_label"] = best_passage_label
            citation_dict["passage_score"] = best_passage_score
        citation_data_list[citation_index] = citation_dict
        citations_data += citation_data_list

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
    display_text, citation_data_list = post_process_output_text(display_text, reference_id_list, citation_map_data)
    citations_data += citation_data_list
    yield display_text, citations_data
    time.sleep(0.1)


def load_example(file_path=""):
    example_text = ""
    with open(file_path, "r") as fi:
        for line in fi.readlines():
            example_text += line
    return example_text


if __name__ == "__main__":
    create_fulltext_dataset = True # total fulltext: 49605

    if not create_fulltext_dataset:
        model_path = "scholarcopilot_model_v1208/"
        device = torch.device("cuda")
        model, tokenizer = load_model(model_path, device)
    
    meta_data_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    meta_data = load_meta_data(meta_data_path)
    print("meta_data size: ", len(meta_data))
    if create_fulltext_dataset:
        # create_fulltext_corpus_data(meta_data_path)
        update_fulltext_corpus_data("scholarcopilot_data/corpus_data_arxiv_1215_fulltext.jsonl")
        exit(0)
    
    citation_map_data_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    citation_map_data = load_citation_map_data(citation_map_data_path)

    index_dir = "scholarcopilot_data/"
    index, lookup_indices = load_faiss_index(index_dir)
    print("index building finished")


    config = get_config()
    passage_retrieval_models = get_passage_retrieval_models(config)
    print("passage retrieval models loaded")


    citations_data = []
    curr_search_candidates = []

    # starting left-side context for the generation model
    example_path = "scholarcopilot_examples/vlm2vec-example.txt"
    text_input = load_example(example_path)

    print("pre-generation text_input:", text_input)

    gen = stream_generate(text_input, citations_data, passage_retrieval_models, config)
    for t in gen:
        text_input, citations_data = t
    print("text_input:", text_input)
    
    save_results({
        "given generation input": example_path,
        "generated paper": text_input,
        "citations_data": citations_data
    }, config, mode="generation")


