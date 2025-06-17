from datetime import datetime
import tempfile
from scholar_copilot_model import *
import torch
import faiss
import time


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


def check_3_sentence(display_text):
    if display_text.endswith('.'):
        return display_text
    end_index = display_text.rfind('.\n')
    return display_text[: end_index + 1]


def stream_complete_3_sentence(text, citations_data, progress=gr.Progress()):
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
        if sentence_num == 3:
            enough = True
            display_text = curr_yield_text
            display_text = check_3_sentence(display_text)
            break
        time.sleep(0.1)
    curr_prefix_length = len(curr_yield_text)
    while cite_start_hidden_state is not None and not enough:
        retrieved_k_results = retrieve_reference(index, lookup_indices, cite_start_hidden_state, top_k=1)
        reference, curr_index = llm_rerank(retrieved_k_results, meta_data)
        reference_id_list.append(curr_index)
        current_text = current_text + reference
        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
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
            if sentence_num == 3:
                enough = True
                display_text = curr_yield_text
                display_text = check_3_sentence(display_text)
                break
            time.sleep(0.1)
        curr_prefix_length = len(curr_yield_text)
    display_text, citation_data_list = post_process_output_text(display_text, reference_id_list, citation_map_data)
    citations_data += citation_data_list
    yield display_text, citations_data
    time.sleep(0.1)


def stream_generate(text, citations_data, progress=gr.Progress()):
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
        current_text = current_text + reference
        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
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
    # return {
    #     citation_box: gr.Group(visible=True),
    #     citation_checkboxes: gr.CheckboxGroup(
    #         choices=choices,
    #         value=[],
    #     ),
    #     curr_search_candidates: curr_search_candidates
    # }
    return gr.Group(visible=True), gr.CheckboxGroup(choices=choices, value=[]), curr_search_candidates


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
    # citations_data = []
    # curr_search_candidates = []
    citations_checkbox = gr.CheckboxGroup(
        choices=[],
        value=[],
    )
    return "", citations_checkbox, "", [], []


def load_example(file_name=""):
    example_text = ""
    with open(f"src/{file_name}", "r") as fi:
        for line in fi.readlines():
            example_text += line
    return example_text


def load_example_text(choice):
    if choice == "Template":
        return load_example("template.txt")
    elif choice == "Example 1":
        return load_example("mmlu-pro-example.txt")
    elif choice == "Example 2":
        return load_example("harness-example.txt")
    elif choice == "Example 3":
        return load_example("vlm2vec-example.txt")


if __name__ == "__main__":
    model_path = "../model_v1208/"
    device = torch.device("cuda:2")
    model, tokenizer = load_model(model_path, device)
    
    meta_data_path = "../data/corpus_data_arxiv_1215.jsonl"
    meta_data = load_meta_data(meta_data_path)
    
    citation_map_data_path = "../data/corpus_data_arxiv_1215.jsonl"
    citation_map_data = load_citation_map_data(citation_map_data_path)

    index_dir = "../data/"
    index, lookup_indices = load_faiss_index(index_dir)
    print("index building finished")


    curr_search_candidates = []

    # app started here
    citations_data = []
    curr_search_candidates = []
    example_text = load_example("template.txt") # choose by calling load_example_text

    # app inputs
    example_text = "Start writing your academic paper..."

    # app buttons
    text_input, citations_data = stream_complete_3_sentence(text_input, citations_data)

    text_input, citations_data = stream_generate(text_input, citations_data)

    citation_box, citation_checkboxes, curr_search_candidates = search_and_show_citations(text_input)

    text_input = insert_selected_citations(text_input, citation_checkboxes, citations_data, curr_search_candidates)

    text_input, citation_checkboxes, bibtex_display, citations_data, curr_search_candidates = clear_cache(citations_data, curr_search_candidates)

    bibtex_display = update_bibtex(citation_data)

    text_input = load_example_text(example_selector)

