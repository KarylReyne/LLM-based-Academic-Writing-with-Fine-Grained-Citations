from scholarcopilot_model import *
import time
import passage_reranking
from passage_retrieval_interface import *
from dataset_loaders import arxiv_to_corpus_id, load_retrieval_dataset, load_scholarcopilot_metadata_corpus


def split_yield_list(input_text, prefix_length):
    prefix_text = input_text[:prefix_length]
    text = input_text[prefix_length:]
    text_list = text.split(" ")
    return prefix_text, text_list


def stream_generate(text, citations_data, index, lookup_indices, model, tokenizer, retrieval_dataset, sc_metadata_corpus, passages_data_dataset, docs_corpus_id_map, passage_retrieval_models, config, generation_breakpoint=15000, do_passage_retrieval=True, catch_retrieval_fails=True, silent=False, save_passage_records=True):
    sentence_num = 0
    retrieval_fails = 0
    llm_fails = 0
    sc_corpus_id_map = {}
    enough = False
    current_text = text
    current_text = preprocess_input_text(current_text)
    display_text = current_text.replace("<|paper_start|> ", "")
    curr_prefix_length = len(display_text)
    current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, passage_retrieval_models, config, current_text, generation_breakpoint=generation_breakpoint, silent=silent)
    unique_reference_id_list = [] # (arxiv_id, suffix)
    display_text, new_citation_data = replace_citations(
        current_text, unique_reference_id_list, retrieval_dataset, sc_metadata_corpus, docs_corpus_id_map, sc_corpus_id_map, silent=silent
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
        yield curr_yield_text, citations_data, [retrieval_fails, llm_fails]
        time.sleep(0.1)
    curr_prefix_length = len(curr_yield_text)

    while cite_start_hidden_state is not None and not enough:
        retrieval_failed = False
        if config["use_only_passage_retriever"]:
            references, distances = retrieve_passages_with_reasonir(index, lookup_indices, cite_start_hidden_state, passages_data_dataset, config, top_k=config["only_passage_retriever_topk"], silent=silent)
        else:
            retrieved_k_results = retrieve_reference(
                index, lookup_indices, cite_start_hidden_state, config, top_k=config["sc_retriever_topk"], silent=silent
            )
            try:
                references, distances = collect_retrieval_results(retrieved_k_results, retrieval_dataset, docs_corpus_id_map, sc_metadata_corpus, silent=silent)
            except ScholarCopilotRetrievalError as e:
                retrieval_fails += 1
                retrieval_failed = True
                if catch_retrieval_fails:
                    sc_corpus_id, _ = retrieved_k_results[0]
                    best_matching_passage = sc_metadata_corpus[sc_corpus_id]["abstract"]+"<|cite_end|>"
                    best_reference_arxiv_id = sc_metadata_corpus[sc_corpus_id]["paper_id"]
                    # add this id to the sc corpus id map since its not from the docs dataset
                    # so that it can be found later by replace_citations()
                    sc_corpus_id_map[best_reference_arxiv_id] = sc_corpus_id
                    if not silent:
                        print("sc retrieval failed, using best abstract as reference: ", best_matching_passage)
                else:
                    raise e

        # --- BEGIN passage retrieval ---
        if not retrieval_failed:
            start = time.time()
            if do_passage_retrieval:
                try:
                    reranking_results = retrieve_relevant_passages(
                        current_text, references, tokenizer, passage_retrieval_models, config, silent=silent, save_passage_records=save_passage_records
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
                    llm_fails += 1
                    retrieval_failed = True
                    if not silent:
                        print("tex or llm response parsing failed, using abstract as reference: ", best_matching_passage)
                
            elif config["use_only_passage_retriever"]: # reasonir instead of sc retrieval, but no reranking
                best_matching_passage = references[0]["passage"]
                best_passage_label = references[0]["passage_label"]
                best_passage_score = distances[0]
                best_reference_arxiv_id = best_passage_label.split("_")[0]
                best_matching_passage = best_matching_passage+"<|cite_end|>"
                if not silent:
                    print("best matching passage: ", best_matching_passage)
                
            else: # standart sc behaviour
                sc_corpus_id, _ = retrieved_k_results[0]
                best_matching_passage = sc_metadata_corpus[sc_corpus_id]["abstract"]+"<|cite_end|>"
                best_reference_arxiv_id = sc_metadata_corpus[sc_corpus_id]["paper_id"]
                # add this id to the sc corpus id map since its not from the docs dataset
                # so that it can be found later by replace_citations()
                sc_corpus_id_map[best_reference_arxiv_id] = sc_corpus_id
                if not silent:
                    print("best abstract as reference: ", best_matching_passage)
            if not silent:
                print(f"best reference after retrieval: {best_reference_arxiv_id}")
                print("***************Retrieval cost (time): ", time.time() - start)
        # --- END passage retrieval ---

        unique_id_suffix = len(unique_reference_id_list) # this resolves duplicate arxiv_ids in the list of references
        unique_reference_id_list.append((best_reference_arxiv_id, unique_id_suffix)) # (arxiv_id, suffix)

        current_text = current_text + best_matching_passage

        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, passage_retrieval_models, config, current_text, generation_breakpoint=generation_breakpoint, silent=silent)
        display_text, new_citation_data = replace_citations(
            current_text, unique_reference_id_list, retrieval_dataset, sc_metadata_corpus, docs_corpus_id_map, sc_corpus_id_map, silent=silent
        )

        citations_data += new_citation_data

        # add passage retrieval-specific entries to citations data
        if (do_passage_retrieval or config["use_only_passage_retriever"]) and (not retrieval_failed) and len(citations_data) > 0:
            # get the data entry of the newly added citation
            citation_dict = citations_data[-1]
            # check that its the correct entry
            if citation_dict["citation_key"] == f"arxivID-{best_reference_arxiv_id}-{unique_id_suffix}":
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
                if not silent:
                    print("sentence_num: ", sentence_num, "each", each)
            curr_yield_text += " " + each
            yield curr_yield_text, citations_data, [retrieval_fails, llm_fails]
            time.sleep(0.1)
        curr_prefix_length = len(curr_yield_text)

    display_text, new_citation_data = post_process_output_text(
        display_text, unique_reference_id_list, retrieval_dataset, sc_metadata_corpus, docs_corpus_id_map, sc_corpus_id_map, silent=silent
    )
    citations_data += new_citation_data
    yield display_text, citations_data, [retrieval_fails, llm_fails]
    time.sleep(0.1)

