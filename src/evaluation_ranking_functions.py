from passage_retrieval_interface import get_config, get_passage_retrieval_models, retrieve_relevant_passages, save_results, apply_retrieval_context_window
from passage_reranking import InvalidLLMResponseError
from scholarcopilot_model import collect_retrieval_results, load_model, load_faiss_index, single_step_retrieval


def rank_with_scholarcopilot(context, index, lookup_indices, model, tokenizer, config):
    silence = True
    context = apply_retrieval_context_window(context, tokenizer, config)
    retrieved_k_results = single_step_retrieval(context, index, lookup_indices, model, tokenizer, config, silent=silence)
    ranked_corpus_ids = [t[0] for t in retrieved_k_results]
    return ranked_corpus_ids


def rank_with_scholarcopilot_with_passageretrieval(context, retrieval_dataset, arxiv_to_corpus_id_map, index, lookup_indices, model, tokenizer, passage_retrieval_models, config):
    silence = True
    context = apply_retrieval_context_window(context, tokenizer, config)
    retrieved_k_results = single_step_retrieval(context, index, lookup_indices, model, tokenizer, config, silent=silence)
    references, _ = collect_retrieval_results(retrieved_k_results, retrieval_dataset, silent=silence)
    response_failure = False
    try:
        ranked_passages, ranked_passage_labels, ranked_passage_scores, _ = retrieve_relevant_passages(
            context, references, passage_retrieval_models, config, silent=silence, save_passage_records=False
        )
        ranked_corpus_ids = [arxiv_to_corpus_id_map[l.split("_")[0]] for l in ranked_passage_labels]
    except InvalidLLMResponseError:
        ranked_corpus_ids = [t[0] for t in retrieved_k_results]
        response_failure = True
    return ranked_corpus_ids, response_failure