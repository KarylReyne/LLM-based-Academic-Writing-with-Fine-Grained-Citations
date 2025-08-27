import torch

from passage_retrieval_interface import retrieve_relevant_passages, retrieve_passages_with_reasonir
from passage_reranking import InvalidLLMResponseError
from scholarcopilot_model import collect_retrieval_results, single_step_retrieval
from passage_retrieval_instructions import retrieval_instruction_query


def rank_with_scholarcopilot(context, retrieval_dataset, docs_arxiv_to_corpus_id_map, sc_metadata_corpus, index, lookup_indices, model, tokenizer, config):
    silence = True
    # if config["enable_query_context_window"]:
    #     context = apply_retrieval_context_window(context, tokenizer, config)
    retrieved_k_results = single_step_retrieval(context, index, lookup_indices, model, tokenizer, config, silent=silence)
    references, _ = collect_retrieval_results(
        retrieved_k_results, retrieval_dataset, docs_arxiv_to_corpus_id_map, sc_metadata_corpus, silent=silence
    )
    ranked_corpus_ids = [item["corpus_id"] for item in references]
    return ranked_corpus_ids, references


def rank_with_reasonir(context, passage_retrieval_models, index, lookup_indices, passages_data_dataset, docs_arxiv_to_corpus_id_map, config):
    silence = True
    with torch.no_grad():
        cite_rep = passage_retrieval_models["retriever"].encode(
            context,
            instruction=retrieval_instruction_query
        )
    references, _ = retrieve_passages_with_reasonir(index, lookup_indices, cite_rep, passages_data_dataset, config, top_k=config["only_passage_retriever_topk"], silent=silence)
    ranked_corpus_ids = [docs_arxiv_to_corpus_id_map[ref["arxiv_id"]] for ref in references]
    return ranked_corpus_ids, references


def rank_with_passage_retrieval_from_sc_rankings(context, references, docs_arxiv_to_corpus_id_map, tokenizer, passage_retrieval_models, config):
    silence = True
    response_failure = False
    try:
        reranking_results = retrieve_relevant_passages(
            context, references, tokenizer, passage_retrieval_models, config, silent=silence, save_passage_records=False
        )
        ranked_corpus_ids = [docs_arxiv_to_corpus_id_map[l.split("_")[0]] for l in reranking_results["ranked_passage_labels"]]
        return ranked_corpus_ids, response_failure, reranking_results, None
    except InvalidLLMResponseError as err:
        response_failure = True
        ranked_corpus_ids = [ref["corpus_id"] for ref in references]
        return ranked_corpus_ids, response_failure, None, err.args

