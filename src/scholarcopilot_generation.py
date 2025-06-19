from datetime import datetime
import tempfile
from scholarcopilot_model import *
import torch
import faiss
import time
import tarfile

from passage_retrieval_interface import *


def generate_citation(input_text):
    global index
    new_input_text = input_text + " <|cite_start|>"
    new_input = tokenizer(new_input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        new_output = model(
            new_input.input_ids,
            attention_mask=new_input.attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
    cite_rep = new_output.hidden_states[-1][:, -1, :]
    retrieved_k_results = retrieve_reference(index, lookup_indices, cite_rep, top_k=10)
    searched_citations = []
    for each in retrieved_k_results:
        curr_index, distance = each
        print("index", curr_index)
        if curr_index not in meta_data:
            print("index not found in meta_data", curr_index)
            continue
        paper_id = meta_data[curr_index]["paper_id"]
        print("paper_id", paper_id)
        citation_info = citation_map_data[paper_id]
        print("generate_citation citation_info", citation_info)
        searched_citations.append(citation_info)
    return searched_citations


def split_yield_list(input_text, prefix_length):
    prefix_text = input_text[:prefix_length]
    text = input_text[prefix_length:]
    text_list = text.split(" ")
    return prefix_text, text_list


def stream_generate(text, citations_data, retr_tokenizer, retriever, rera_tokenizer, reranker, config):
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
        retrieved_k_results = retrieve_reference(index, lookup_indices, cite_start_hidden_state, top_k=1)
        reference, curr_index = llm_rerank(retrieved_k_results, meta_data)
        reference_id_list.append(curr_index)

        # --- BEGIN passage retrieval ---
        generated_context = current_text
        reference_id = curr_index
        tex_parsing_failed = False
        try:
            best_matching_passage, best_passage_label, best_passage_score = retrieve_relevant_passages(
                generated_context, reference_id, retr_tokenizer, retriever, rera_tokenizer, reranker, config
            )
            best_matching_passage = best_matching_passage+"<|cite_end|>"
            print("best matching passage: ", best_matching_passage)
        except tarfile.ReadError:
            tex_parsing_failed = True
            best_matching_passage = reference # default to standart ScholarCopilot if parsing failed
            print("latex parsing failed, using reference: ", best_matching_passage)
        # --- END passage retrieval ---

        # current_text = current_text + reference
        current_text = current_text + best_matching_passage

        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
        display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)

        # citations_data += citation_data_list
        ids = [d["paper_id"] for d in citation_data_list]
        citation_index = ids.index(reference_id)
        assert citation_index == len(citation_data_list)-1
        citation_dict = citation_data_list[citation_index]
        if tex_parsing_failed:
            citation_dict["matched_passage"] = "<|tex_parsing_failed|>"
            citation_dict["passage_label"] = "<|tex_parsing_failed|>"
            citation_dict["passage_score"] = "<|tex_parsing_failed|>"
        else:
            citation_dict["matched_passage"] = best_matching_passage.rstrip("<|cite_end|>")
            citation_dict["passage_label"] = best_passage_label
            citation_dict["passage_score"] = best_passage_score
        citation_data_list[0] = citation_dict
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


def format_citation(citation_key, url):
    total_length = 150
    citation_length = len(citation_key)
    url_length = len(url)
    if citation_length > 110:
        citation_key = citation_key[:105] + "...  "
        citation_length = 110
    return citation_key + " " * (total_length - citation_length - url_length) + url


def search_and_show_citations(input_text):
    curr_citations_data = generate_citation(input_text)
    curr_search_candidates = curr_citations_data
    choices = []
    for cit in curr_citations_data:
        # print("cit.keys()", list(cit.keys()))
        paper_id = cit["paper_id"]
        citation_key = cit["citation_key"]
        title = cit["title"].replace("\n", " ").replace("  ", " ")
        url = f" (https://arxiv.org/abs/{paper_id})"
        item = format_citation(citation_key + ": " + title, url)
        # print("item", item)
        choices.append(item)
    return curr_search_candidates


def insert_selected_citations(text, selected_citations, citations_data, curr_search_candidates):
    if not selected_citations:
        return text

    selected_citations = [each.split(": ")[0] for each in selected_citations]
    citations = ", ".join(selected_citations)
    new_text = text + " \\cite{" + citations + "}"
    for each_candidate in curr_search_candidates:
        if each_candidate["citation_key"] in selected_citations:
            citations_data.append(each_candidate)
    return new_text


def update_bibtex(citations_data):
    # print("citations_data", citations_data)
    if not citations_data:
        return None  # 如果没有引用历史，返回None

    bibtex_entries = []
    for cit in citations_data:
        if cit["bibtex"] not in bibtex_entries:
            bibtex_entries.append(cit["bibtex"])
    content = "\n\n".join(bibtex_entries)
    return content


def clear_cache(citations_data, curr_search_candidates):
    return "", False, "", [], []


def load_example(file_path=""):
    example_text = ""
    with open(file_path, "r") as fi:
        for line in fi.readlines():
            example_text += line
    return example_text


if __name__ == "__main__":
    model_path = "scholarcopilot_model_v1208/"
    device = torch.device("cuda")
    model, tokenizer = load_model(model_path, device)
    
    meta_data_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    meta_data = load_meta_data(meta_data_path)
    
    citation_map_data_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    citation_map_data = load_citation_map_data(citation_map_data_path)

    index_dir = "scholarcopilot_data/"
    index, lookup_indices = load_faiss_index(index_dir)
    print("index building finished")


    config = get_config()
    retr_tokenizer, retriever, rera_tokenizer, reranker = get_passage_retrieval_models(config)
    print("passage retrieval models loaded")


    citations_data = []
    curr_search_candidates = []

    # starting left-side context for the generation model
    example_path = "scholarcopilot_examples/vlm2vec-example.txt"
    text_input = load_example(example_path)

    print("pre-generation text_input:", text_input)

    gen = stream_generate(text_input, citations_data, retr_tokenizer, retriever, rera_tokenizer, reranker, config)
    for out in gen:
        text_input, citations_data = out
    print("text_input:", text_input)
    
    save_results({
        "given generation input": example_path,
        "generated paper": text_input,
        "citations_data": citations_data
    }, config, mode="generation")


